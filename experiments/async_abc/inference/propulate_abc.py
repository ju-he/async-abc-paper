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
import hashlib
import json
import logging
import math
import random
import shutil
import time
from contextlib import contextmanager
from typing import Callable, Dict, List

import numpy as np

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from ..utils.mpi import get_rank, is_root_rank

Propulator = None
ABCPMC = None
logger = logging.getLogger(__name__)


def _make_propulate_comm():
    """Return a per-run communicator to avoid cross-run message reuse."""
    try:
        from mpi4py import MPI
    except Exception:
        return None

    try:
        if MPI.COMM_WORLD.Get_size() <= 1:
            return None
        MPI.COMM_WORLD.Barrier()  # sync all ranks + drain stale messages before Dup
        return MPI.COMM_WORLD.Dup()
    except Exception:
        return None


def _free_propulate_comm(comm) -> None:
    """Best-effort communicator cleanup after a completed Propulate run."""
    if comm is None:
        return
    try:
        comm.Free()
    except Exception:
        pass


def _cleanup_propulate_intra_requests(propulator) -> int:
    """Prune completed intra-island nonblocking sends.

    Without periodic pruning, ``intra_requests`` grows unboundedly during a
    wall-time-limited run.  High-throughput configurations (e.g. ablation with
    k=10) can accumulate tens of thousands of outstanding ``isend`` requests,
    exhausting ParaStationMPI/pscom internal resources and causing ``isend`` to
    block.  Calling ``Testsome`` each iteration retires completed sends so the
    outstanding count stays bounded.
    """
    requests = getattr(propulator, "intra_requests", None)
    if not requests:
        return 0
    try:
        from mpi4py import MPI as _MPI
        indices = _MPI.Request.Testsome(requests)
        if indices is not None:
            # Remove completed entries in reverse order to preserve indices.
            buffers = getattr(propulator, "intra_buffers", None)
            for idx in sorted(indices, reverse=True):
                del requests[idx]
                if buffers is not None and idx < len(buffers):
                    del buffers[idx]
        return len(requests)
    except Exception:
        logger.debug("Propulate intra-request cleanup failed", exc_info=True)
        return len(requests)


def _propulate_world_size() -> int:
    """Return the active Propulate world size, or 1 when MPI is unavailable."""
    try:
        from mpi4py import MPI
    except Exception:
        return 1

    try:
        return max(1, int(MPI.COMM_WORLD.Get_size()))
    except Exception:
        return 1


def _effective_generation_budget(max_sims: int, inference_cfg: Dict) -> int:
    """Return the Propulate generation count for this run.

    In normal runs we preserve the historical behavior. In test mode we treat
    ``max_simulations`` as a total budget across ranks so MPI smoke tests do not
    multiply the requested budget by the worker count.
    """
    if inference_cfg.get("propulate_budget_mode") == "total_simulations":
        return max(1, math.ceil(int(max_sims) / _propulate_world_size()))
    if not inference_cfg.get("test_mode"):
        return int(max_sims)
    return max(1, math.ceil(int(max_sims) / _propulate_world_size()))


def _ensure_propulate_imports() -> None:
    """Resolve propulate from the active Python environment."""
    global Propulator, ABCPMC
    if Propulator is not None and ABCPMC is not None:
        return

    try:
        from propulate import Propulator as _Propulator
        from propulate.propagators.abcpmc import ABCPMC as _ABCPMC
    except ImportError as env_exc:
        raise ImportError(
            "The async_propulate_abc method requires 'propulate'. "
            "Install it in the active environment."
        ) from env_exc

    if Propulator is None:
        Propulator = _Propulator
    if ABCPMC is None:
        ABCPMC = _ABCPMC


def _stable_seed(*parts: object) -> int:
    """Return a stable 31-bit seed derived from structured inputs."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def _param_key(params: Dict[str, float]) -> tuple[tuple[str, float], ...]:
    """Return a stable, rounded parameter key for deterministic seeding."""
    return tuple(sorted((key, round(float(value), 15)) for key, value in params.items()))


def _eval_seed(
    run_seed: int,
    mpi_rank: int,
    generation: int,
    params: Dict[str, float],
) -> int:
    """Deterministic per-evaluation seed derived from run, rank, and params."""
    return _stable_seed(run_seed, mpi_rank, generation, _param_key(params))


def _individual_params(ind, limits: Dict) -> Dict[str, float]:
    """Extract parameter values from a Propulate individual with a clear error."""
    params: Dict[str, float] = {}
    for key in limits:
        try:
            params[key] = float(ind[key])
        except Exception as exc:
            available_keys = sorted(str(k) for k in getattr(ind, "keys", lambda: [])())
            raise RuntimeError(
                "Propulate returned an individual incompatible with the current "
                f"limits. Expected keys {sorted(limits)}, got {available_keys}. "
                "This usually indicates checkpoint reuse or MPI message "
                "cross-contamination between sequential Propulate runs."
            ) from exc
    return params


@contextmanager
def _suppress_propulate_info_logs():
    """Temporarily silence Propulate INFO logs so wrapper progress stays primary."""
    propulate_logger = logging.getLogger("propulate")
    original_level = propulate_logger.level
    propulate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        propulate_logger.setLevel(original_level)


def _resolve_max_wall_time_s(inference_cfg: Dict) -> float | None:
    """Return the configured wall-time cap, if any."""
    max_wall_time_s = inference_cfg.get("max_wall_time_s")
    if max_wall_time_s in (None, ""):
        return None
    return float(max_wall_time_s)


def _prepare_checkpoint_dir(checkpoint_dir, *, inference_cfg: Dict) -> None:
    """Reset stale checkpoints for test runs before Propulate starts."""
    if bool(inference_cfg.get("test_mode", False)) and is_root_rank() and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        from mpi4py import MPI as _MPI

        if _MPI.COMM_WORLD.Get_size() > 1:
            _MPI.COMM_WORLD.Barrier()
    except Exception:
        pass



def _wall_time_exceeded(run_start: float, max_wall_time_s: float) -> bool:
    """Return whether this rank has locally exceeded the wall-time budget."""
    return (time.time() - float(run_start)) >= float(max_wall_time_s)


def _propulate_with_wall_time_limit(
    propulator,
    *,
    run_start: float,
    max_wall_time_s: float,
    logging_interval: int,
    debug: int = 0,
) -> None:
    """Run Propulate until this worker's local clock exceeds *max_wall_time_s*.

    Each rank checks its own clock independently — no collective operations in
    the hot loop — so workers remain fully asynchronous.  The post-loop barriers
    ensure population consistency after all ranks have exited.  Results that
    completed after the deadline are filtered out by the caller
    (``run_propulate_abc``), giving semantics identical to a hard job abort.
    """
    try:
        from mpi4py import MPI
    except Exception:
        MPI = None

    worker_sub_comm = getattr(propulator, "worker_sub_comm", None)
    if MPI is not None and worker_sub_comm not in (None, MPI.COMM_SELF):
        propulator.generation = worker_sub_comm.bcast(propulator.generation, root=0)

    propulate_comm = getattr(propulator, "propulate_comm", None)
    if propulate_comm is None:
        while propulator.generations <= -1 or propulator.generation < propulator.generations:
            if _wall_time_exceeded(run_start, max_wall_time_s):
                break
            propulator._evaluate_individual()
            propulator.generation += 1
        return

    if getattr(propulator, "island_comm", None) is not None and propulator.island_comm.rank == 0:
        logger.debug(
            "Running Propulate with a %.3fs local wall-time cap",
            float(max_wall_time_s),
        )

    dump = bool(getattr(getattr(propulator, "island_comm", None), "rank", None) == 0)
    propulate_comm.barrier()

    while propulator.generations <= -1 or propulator.generation < propulator.generations:
        if _wall_time_exceeded(run_start, max_wall_time_s):
            break

        if propulator.generation % int(logging_interval) == 0:
            logger.debug("Propulate generation=%s", propulator.generation)

        propulator._evaluate_individual()
        propulator._receive_intra_island_individuals()
        _cleanup_propulate_intra_requests(propulator)

        if dump:
            propulator._dump_checkpoint()
        dump = propulator._determine_worker_dumping_next()
        propulator.generation += 1

    propulate_comm.barrier()
    # Drain remaining intra-island messages and wait for all outgoing sends.
    # ALL ranks must participate in the drain loop collectively.  If only ranks
    # with pending sends loop (the previous approach), ranks whose send lists
    # were already pruned escape to a barrier while senders still need them to
    # post matching recvs — deadlocking under MPI rendezvous mode.
    intra_reqs = getattr(propulator, "intra_requests", None)
    from mpi4py import MPI as _MPI

    while True:
        propulator._receive_intra_island_individuals()
        try:
            sends_done = not intra_reqs or _MPI.Request.Testall(intra_reqs)
        except TypeError:
            sends_done = True
        local_done = 1 if sends_done else 0
        global_done = np.zeros(1, dtype=np.int32)
        propulate_comm.Allreduce(
            np.array([local_done], dtype=np.int32), global_done, op=_MPI.MIN,
        )
        if global_done[0]:
            break

    if intra_reqs:
        propulator.intra_requests.clear()
        buffers = getattr(propulator, "intra_buffers", None)
        if buffers is not None:
            buffers.clear()
    propulate_comm.barrier()

    island_comm = getattr(propulator, "island_comm", None)
    if island_comm is not None and island_comm.rank == 0:
        propulator._dump_final_checkpoint()
    propulate_comm.barrier()
    propulator._determine_worker_dumping_next()
    propulate_comm.barrier()


def run_propulate_abc(
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    progress=None,
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
    generation_budget = _effective_generation_budget(max_sims, inference_cfg)
    max_wall_time_s = _resolve_max_wall_time_s(inference_cfg)

    # When a wall-time cap is set it should be the binding stopping criterion,
    # not the generation budget.  Use -1 (unlimited) so workers keep running
    # until the local time check fires; the post-hoc filter discards anything
    # that completed after the deadline.
    if max_wall_time_s is not None:
        generation_budget = -1
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

    mpi_rank = get_rank()

    propagator = ABCPMC(
        limits=limits,
        perturbation_scale=perturbation_scale,
        k=k,
        tol=tol_init,
        scheduler_type=scheduler_type,
        rng=random.Random(_stable_seed(seed, "propagator", mpi_rank)),
        **scheduler_kwargs,
    )

    run_start = time.time()
    eval_count = 0
    # Propulate's loss_fn receives an Individual (dict-like).
    # We extract param values and forward to the benchmark simulator.
    def loss_fn(ind) -> float:
        nonlocal eval_count
        params = _individual_params(ind, limits)
        sim_seed = _eval_seed(seed, mpi_rank, int(ind.generation), params)
        loss = float(simulate_fn(params, seed=sim_seed))
        eval_count += 1
        if progress is not None:
            progress.update(evaluations=eval_count)
        return loss

    # Each (replicate, seed) gets its own checkpoint dir to prevent cross-contamination.
    # An optional _checkpoint_tag in inference_cfg further qualifies the path so that
    # callers such as the ablation runner (which share output_dir across variants) do
    # not accidentally resume a checkpoint from a different configuration variant.
    _tag = inference_cfg.get("_checkpoint_tag", "")
    _tag_suffix = f"__{_tag}" if _tag else ""
    checkpoint_dir = output_dir.logs / f"propulate_rep{replicate}_seed{seed}{_tag_suffix}"
    _prepare_checkpoint_dir(checkpoint_dir, inference_cfg=inference_cfg)

    propulate_comm = _make_propulate_comm()
    propulator_kwargs = {}
    if propulate_comm is not None:
        try:
            from mpi4py import MPI
            propulator_kwargs = {
                "island_comm": propulate_comm,
                "propulate_comm": propulate_comm,
                "worker_sub_comm": MPI.COMM_SELF,
            }
        except Exception:
            propulator_kwargs = {}

    propulator = Propulator(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=random.Random(_stable_seed(seed, "propulator", mpi_rank)),
        generations=generation_budget,
        checkpoint_path=checkpoint_dir,
        **propulator_kwargs,
    )

    logging_interval = max(1, generation_budget + 1)
    try:
        with _suppress_propulate_info_logs():
            if max_wall_time_s is not None:
                _propulate_with_wall_time_limit(
                    propulator,
                    run_start=run_start,
                    max_wall_time_s=max_wall_time_s,
                    logging_interval=logging_interval,
                    debug=0,
                )
            else:
                propulator.propulate(logging_interval=logging_interval, debug=0)
    finally:
        # Unconditional cleanup: run whether the run completed, timed out, or
        # raised.  Any pending intra-island sends left here indicate a bug in
        # the wall-time cleanup path (they should have been Waitall'd above).
        pending = _cleanup_propulate_intra_requests(propulator)
        if pending > 0:
            logger.error(
                "Propulate run ended with %d pending intra-island send requests; "
                "forcing communicator free. This indicates a cleanup bug.",
                pending,
            )
        _free_propulate_comm(propulate_comm)
        # Synchronize all ranks before the next replicate or runner phase starts.
        # Without this barrier, lagging ranks can still be in transport teardown
        # when the next replicate calls _make_propulate_comm() → COMM_WORLD.Barrier(),
        # causing a hang or ParaStation pscom assertion on the following Dup().
        try:
            from mpi4py import MPI as _MPI
            if _MPI.COMM_WORLD.Get_size() > 1:
                _MPI.COMM_WORLD.Barrier()
        except Exception:
            pass

    # Sort by completion time so the record order reflects the observable
    # event stream rather than generation assignment alone.
    population = sorted(
        propulator.population,
        key=lambda ind: (
            float(ind.evaltime) if hasattr(ind, "evaltime") and ind.evaltime is not None else float("inf"),
            int(ind.generation) if getattr(ind, "generation", None) is not None else 0,
            int(getattr(ind, "rank", 0) or 0),
        ),
    )
    records: List[ParticleRecord] = []
    current_tolerance = float(tol_init)
    for step, ind in enumerate(population, start=1):
        params = _individual_params(ind, limits)
        weight = float(ind.weight) if ind.weight is not None else None
        if ind.tolerance is not None:
            current_tolerance = min(current_tolerance, float(ind.tolerance))
            tolerance = current_tolerance
        else:
            tolerance = current_tolerance
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
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=step,
        ))

    # Discard results that completed after the wall-time deadline.  This gives
    # semantics identical to a hard job abort: every rank runs independently and
    # only work finished before the deadline counts.
    if max_wall_time_s is not None:
        records = [
            r for r in records
            if r.sim_end_time is None or r.sim_end_time <= max_wall_time_s
        ]

    if progress is not None:
        progress.finish(evaluations=eval_count, records=len(records))
    return records
