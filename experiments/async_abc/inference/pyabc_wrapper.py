"""pyABC inference wrapper (optional dependency).

If ``pyabc`` is not installed, :func:`run_pyabc_smc` raises ``ImportError``
with installation instructions.  The function is still importable so that the
method registry can reference it unconditionally.
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
    build_pyabc_sampler,
    resolve_pyabc_parallel_backend,
    resolve_pyabc_worker_count,
)

from ._pyabc_common import db_suffix as _db_suffix, prepare_db_path as _prepare_db_path

logger = logging.getLogger(__name__)


class _FutureTracker:
    """Thin executor wrapper that tracks submitted futures for pre-shutdown draining.

    pyABC's EPSMixin removes futures from its internal tracking list and calls
    ``future.cancel()`` once enough accepted particles have been collected.  For
    mpi4py MPI futures ``cancel()`` is a no-op on already-running tasks: workers
    complete execution and then block in ``comm.send()`` waiting to deliver a
    result that nobody will receive.  When the ``MPICommExecutor`` context exits,
    its ``shutdown(wait=True)`` deadlocks because the manager thread is stuck in
    a blocking ``probe()`` waiting for those orphan results.

    By tracking every submitted future and calling ``concurrent.futures.wait()``
    before the context exits, we ensure all workers can deliver their results and
    the manager thread drains cleanly — ``pool.join()`` then completes instantly.
    """

    def __init__(self, inner):
        self._inner = inner
        self._submitted: list = []

    def submit(self, fn, /, *args, **kwargs):
        f = self._inner.submit(fn, *args, **kwargs)
        self._submitted.append(f)
        return f

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def drain(self) -> None:
        """Wait for any futures that are still running after sampling returned."""
        from concurrent.futures import wait as _wait
        pending = [f for f in self._submitted if not f.done()]
        if pending:
            logger.debug(
                "[pyabc] draining %d orphan futures before MPICommExecutor shutdown",
                len(pending),
            )
            _wait(pending)
            logger.debug("[pyabc] orphan futures drained")


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
    checkpoint_tag: str = "",
    max_wall_time_s: float | None = None,
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

    _distance_cache: Dict = {}
    eval_count = 0

    trace_dir = output_dir.logs / f"pyabc_smc_rep{replicate}_seed{seed}{_db_suffix(checkpoint_tag)}_attempts"
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
        method_name="pyabc_smc",
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
    logger.info(
        "[pyabc] abc.run() starting: replicate=%d seed=%d max_sims=%d",
        replicate, seed, max_sims,
    )
    history = abc.run(
        minimum_epsilon=tol_init * 0.01,
        max_total_nr_simulations=max_sims,
        max_walltime=(
            timedelta(seconds=float(max_wall_time_s))
            if max_wall_time_s is not None
            else None
        ),
    )
    logger.info(
        "[pyabc] abc.run() finished: replicate=%d seed=%d max_t=%d",
        replicate, seed, history.max_t,
    )

    observable = history_observable_frame(history, run_start)

    # Collect records from the pyABC history
    logger.info(
        "[pyabc] loading attempt events: replicate=%d seed=%d trace_dir=%s",
        replicate, seed, trace_dir,
    )
    attempt_events = load_attempt_events(trace_dir, run_start_abs=run_start)
    logger.info(
        "[pyabc] loaded %d attempt events: replicate=%d seed=%d",
        len(attempt_events), replicate, seed,
    )
    loss_by_param_key = {
        str(event["param_key"]): float(event["loss"])
        for event in attempt_events
    }

    logger.info("[pyabc] building records: replicate=%d seed=%d", replicate, seed)
    records: List[ParticleRecord] = []
    records.extend(
        attempt_records_from_events(
            attempt_events,
            method_name="pyabc_smc",
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

    logger.info(
        "[pyabc] records built: %d total (%d attempts, %d population_particles) "
        "replicate=%d seed=%d",
        len(records),
        sum(1 for r in records if r.record_kind == "simulation_attempt"),
        sum(1 for r in records if r.record_kind == "population_particle"),
        replicate, seed,
    )
    if progress is not None:
        # Use len(attempt_events) rather than eval_count: under MPI the pyabc_model
        # callback executes on worker ranks so root's eval_count stays at 0 while
        # attempt_events captures all traced work from every rank.
        progress.finish(
            simulations=len(attempt_events),
            generations=history.max_t + 1,
            records=len(records),
        )
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
    max_wall_time_s = inference_cfg.get("max_wall_time_s")
    max_wall_time_s = None if max_wall_time_s in (None, "") else float(max_wall_time_s)
    parallel_backend = resolve_pyabc_parallel_backend(
        inference_cfg,
        method_name="pyabc_smc",
        simulate_fn=simulate_fn,
    )
    checkpoint_tag = str(inference_cfg.get("_checkpoint_tag", ""))
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
            tracker = _FutureTracker(executor)
            sampler = build_pyabc_sampler(
                n_procs,
                parallel_backend,
                cfuture_executor=tracker,
            )
            result = _run_pyabc_smc_with_sampler(
                sampler=sampler,
                simulate_fn=simulate_fn,
                limits=limits,
                max_sims=max_sims,
                k=k,
                tol_init=tol_init,
                output_dir=output_dir,
                replicate=replicate,
                seed=seed,
                checkpoint_tag=checkpoint_tag,
                max_wall_time_s=max_wall_time_s,
                progress=progress,
            )
            tracker.drain()
            return result

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
        checkpoint_tag=checkpoint_tag,
        max_wall_time_s=max_wall_time_s,
        progress=progress,
    )
