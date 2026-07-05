"""Propulate-ABC inference wrapper.

Wraps the existing ``ABCPMC`` propagator from ``propulate`` and runs it via
``Propulator`` for a fixed number of simulations.  After the run, all
evaluated individuals are converted to :class:`~async_abc.io.records.ParticleRecord`
objects.

The loss function passed to Propulate receives a ``propulate.Individual`` which
behaves like a dict (``ind["mu"]`` etc.).  Internally, the benchmark's
``simulate(params, seed)`` is called with a per-evaluation seed derived from
the run seed and the individual's generation counter.

Wall-time semantics
-------------------
When ``max_wall_time_s`` is configured each MPI rank polls its own local
clock between evaluations and exits independently — *first-rank-hit*, not
collective. After the loop the post-loop barriers synchronise the
population. Individuals whose evaluation completes after the deadline are
filtered out by ``run_propulate_abc`` so the produced records have a hard
end-of-budget cap matching the pyABC and rejection-ABC paths.
"""
import atexit
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

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
    """Best-effort communicator cleanup after a completed Propulate run.

    On ParaStation MPI at ≥48 ranks, ``MPI_Comm_free`` can take 30+ seconds
    or hang outright in ``pscom_close`` (W3.4). Operators can opt out of
    the Free() call by setting ``PROPULATE_SKIP_DISCONNECT=1`` in the job
    environment; the communicator is then leaked and reclaimed when the
    Python interpreter exits, which is acceptable for batch runs and
    avoids the teardown hang.
    """
    if comm is None:
        return
    if os.environ.get("PROPULATE_SKIP_DISCONNECT", "").strip() in ("1", "true", "yes"):
        logger.debug(
            "PROPULATE_SKIP_DISCONNECT set; skipping MPI_Comm_free on the "
            "Propulate run communicator."
        )
        return
    try:
        comm.Free()
    except Exception:
        pass


# Max bounded (recv + Testsome) rounds when applying send backpressure, so the
# loop can never spin forever even if traffic momentarily stalls. Each round is
# non-blocking and makes progress (drains incoming so peers can retire our sends),
# so the cap is reached well within this bound in practice.
_BACKPRESSURE_MAX_ROUNDS = 1000


def _cleanup_propulate_intra_requests(
    propulator, *, max_inflight=None, drain_recv=None, _testsome=None
) -> int:
    """Prune completed intra-island nonblocking sends, with optional backpressure.

    Without pruning, ``intra_requests`` grows unboundedly during a wall-time-
    limited run.  ``Testsome`` retires *already-completed* sends each iteration —
    but it never blocks, so when peers' receives lag (high rank counts: 96 ranks
    post 95 ``isend``s per evaluation) the outstanding set still grows without
    bound and exhausts ParaStation pscom's per-connection resources, crashing a
    rank mid-run (segfault on UCX, socket drop on TCP — transport-independent).

    With ``max_inflight`` set, this additionally applies **backpressure**: while
    the outstanding count exceeds the cap, it runs bounded rounds of (drain
    incoming via ``drain_recv`` → ``Testsome``-retire our completed sends).
    Draining incoming FIRST lets peers progress and receive our sends, so the
    retire makes progress — all non-blocking, so it cannot deadlock even if every
    rank backpressures at once.  Results are unchanged; only the send pacing is.
    """
    requests = getattr(propulator, "intra_requests", None)
    if not requests:
        return 0
    try:
        if _testsome is None:
            from mpi4py import MPI as _MPI

            _testsome = _MPI.Request.Testsome

        buffers = getattr(propulator, "intra_buffers", None)

        def _retire(indices) -> int:
            if not indices:
                return 0
            for idx in sorted(indices, reverse=True):
                del requests[idx]
                if buffers is not None and idx < len(buffers):
                    del buffers[idx]
            return len(indices)

        _retire(_testsome(requests))

        if max_inflight and len(requests) > int(max_inflight):
            cap = int(max_inflight)
            for _ in range(_BACKPRESSURE_MAX_ROUNDS):
                if drain_recv is not None:
                    drain_recv()  # receive incoming so peers can retire our sends
                _retire(_testsome(requests))
                if len(requests) <= cap:
                    break
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


def _comm_world_is_root() -> bool:
    """Return whether this process is COMM_WORLD rank 0 (or single-process)."""
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank()) == 0
    except Exception:  # noqa: BLE001 - single process / no mpi4py
        return True


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

    # Cap on outstanding intra-island isends (0 disables). Bounds ParaStation pscom
    # per-connection resource use so a worker cannot crash mid-run under the 95-way
    # fan-out at >=2-node scale (see .plans/bug-fixes). 4096 ~= 43/peer at 96 ranks.
    max_inflight_sends = int(os.environ.get("PROPULATE_MAX_INFLIGHT_SENDS", "4096"))

    while propulator.generations <= -1 or propulator.generation < propulator.generations:
        if _wall_time_exceeded(run_start, max_wall_time_s):
            break

        if propulator.generation % int(logging_interval) == 0:
            logger.debug("Propulate generation=%s", propulator.generation)

        propulator._evaluate_individual()
        propulator._receive_intra_island_individuals()
        _cleanup_propulate_intra_requests(
            propulator,
            max_inflight=max_inflight_sends,
            drain_recv=propulator._receive_intra_island_individuals,
        )

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
    # Reuse the guarded MPI import from the top of this function rather than
    # re-importing at function scope. Without mpi4py installed (e.g. unit-test
    # CI without an MPI stack), MPI is None and the rendezvous-safe collective
    # drain is unnecessary — fake comms and single-process runs both fall
    # through to the buffer clear below.
    _MPI = MPI

    # Bound the collective drain. At very high message volume (large k, many
    # ranks) the outstanding intra-island isends may never fully drain over
    # ParaStation pscom, so the unbounded loop spins forever and SLURM force-kills
    # the step (observed: k>=192 at 48/96 ranks hang here ~6s after the wall-time
    # loop ends — BEFORE _free_propulate_comm, which is why PROPULATE_SKIP_DISCONNECT
    # did not help). Every rank shares the same deadline (run_start + max_wall_time_s
    # + grace), so once it passes all ranks report "done" and the Allreduce(MIN)
    # breaks the loop collectively. Leftover messages are reclaimed at process exit —
    # immediate for the one-combo-per-process scaling jobs. Override via
    # PROPULATE_DRAIN_TIMEOUT_S (seconds; default 120).
    drain_grace_s = float(os.environ.get("PROPULATE_DRAIN_TIMEOUT_S", "120"))
    drain_deadline = float(run_start) + float(max_wall_time_s) + drain_grace_s
    drain_timed_out = False

    while _MPI is not None:
        propulator._receive_intra_island_individuals()
        try:
            sends_done = not intra_reqs or _MPI.Request.Testall(intra_reqs)
        except TypeError:
            sends_done = True
        if not sends_done and time.time() >= drain_deadline:
            sends_done = True
            drain_timed_out = True
        local_done = 1 if sends_done else 0
        global_done = np.zeros(1, dtype=np.int32)
        propulate_comm.Allreduce(
            np.array([local_done], dtype=np.int32), global_done, op=_MPI.MIN,
        )
        if global_done[0]:
            # Final drain: consume any messages that arrived during the Allreduce.
            # At this point all sends are globally complete, so no new messages
            # will be generated.  Any remaining buffered receives must already
            # have been handed to MPI by their senders, and this pass collects
            # them before the communicator is freed.  Without this extra pass,
            # messages arriving in the send-complete→Allreduce window are left
            # in propulate_comm's buffer when MPI_Comm_free is called, which
            # corrupts pscom state and causes MPI_Mrecv failures in the next
            # method's CommWorldMap workers.
            propulator._receive_intra_island_individuals()
            break

    if drain_timed_out:
        logger.warning(
            "Propulate post-loop intra-island drain exceeded its %.0fs deadline with "
            "pending sends; proceeding to teardown (leftover messages reclaimed at "
            "process exit). Override via PROPULATE_DRAIN_TIMEOUT_S.",
            drain_grace_s,
        )

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


class _PhaseTimingPropagator:
    """Opt-in (env ``ASYNC_ABC_PHASE_TIMING=1``) timing wrapper around a propagator.

    Accumulates the wall-clock spent inside the propagator's ``__call__`` (the
    per-arrival proposal reconstruction + AMIS-snapshot importance weight) per MPI
    rank and, at process exit, writes one row ``{rank, n_calls, proposal_s, wall_s}``.
    Combined with the per-eval simulator time already in ``raw_results``, this lets the
    strong-scaling decomposition attribute wall-clock to simulator vs proposal vs
    coordination/idle (``comm/idle = wall - simulator - proposal``). Transparent to
    Propulate: it forwards ``set_worker_context``/``extract_posterior`` and delegates
    every other attribute to the wrapped propagator; only ``__call__`` is timed.
    """

    def __init__(self, inner, *, data_dir, tag, replicate):
        self._inner = inner
        self.total_proposal_s = 0.0
        self.n_calls = 0
        self._rank = -1
        self._t0 = time.perf_counter()
        self._data_dir = data_dir
        self._tag = tag or "untagged"
        self._replicate = replicate
        atexit.register(self._flush)

    def set_worker_context(self, rank, size, *args, **kwargs):
        self._rank = int(rank)
        return self._inner.set_worker_context(rank, size, *args, **kwargs)

    def extract_posterior(self, *args, **kwargs):
        return self._inner.extract_posterior(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        _t = time.perf_counter()
        out = self._inner(*args, **kwargs)
        self.total_proposal_s += time.perf_counter() - _t
        self.n_calls += 1
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _flush(self):
        if self.n_calls == 0:
            return
        wall = time.perf_counter() - self._t0
        path = os.path.join(
            str(self._data_dir),
            f"phase_timing_{self._tag}_rep{self._replicate}_rank{self._rank}.csv",
        )
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "replicate", "rank", "n_calls", "proposal_s", "wall_s"])
            w.writerow([self._tag, self._replicate, self._rank, self.n_calls,
                        round(self.total_proposal_s, 5), round(wall, 5)])


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
        ``scheduler_type``, ``perturbation_scale``, ``kernel``
        (``"hard"`` | ``"gaussian"`` | ``"epanechnikov"``, default ``"hard"``),
        ``amis_snapshots`` (default 0), ``amis_interval``.
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
    # Smooth-kernel / AMIS configuration (paper A+D track). Defaults preserve
    # the legacy hard-threshold behaviour for backwards compatibility.
    kernel = inference_cfg.get("kernel", "hard")
    amis_snapshots = int(inference_cfg.get("amis_snapshots", 0))
    amis_interval_cfg = inference_cfg.get("amis_interval")
    # Retroactive AMIS posterior reweighting (extract_posterior) is the reported
    # estimator for posterior-quality experiments (SBC etc.), but it costs
    # O(n_history * amis_snapshots * k) on the post-run analysis path and is NOT
    # bounded by the inference wall-time. On cheap-simulator scaling sweeps the
    # history reaches ~1e6 individuals, where at k=1000 this is 10-16 min of
    # single-threaded NumPy per combo — it overruns the SLURM wall clock and
    # looks like a post-teardown MPI hang. Experiments that do not consume
    # ``posterior_weight`` (the throughput/scaling sweeps) set this False; the
    # weights remain recomputable offline from the saved history if ever needed.
    compute_posterior_weights = bool(inference_cfg.get("compute_posterior_weights", True))
    # Pass extra scheduler kwargs if present
    scheduler_kwargs = {}
    for key in ("percentile", "decay_factor", "low_rate", "high_rate",
                "shrink_factor", "expand_factor"):
        if key in inference_cfg:
            scheduler_kwargs[key] = inference_cfg[key]

    mpi_rank = get_rank()

    abcpmc_kwargs = dict(
        limits=limits,
        perturbation_scale=perturbation_scale,
        k=k,
        tol=tol_init,
        scheduler_type=scheduler_type,
        kernel=kernel,
        amis_snapshots=amis_snapshots,
        rng=random.Random(_stable_seed(seed, "propagator", mpi_rank)),
        **scheduler_kwargs,
    )
    if amis_interval_cfg is not None:
        abcpmc_kwargs["amis_interval"] = int(amis_interval_cfg)
    # One config key governs both arms of the matched comparison: the async
    # side's kernel-aware scheduler and the baseline's matched epsilon
    # (make_matched_epsilon) read the same ESS-retention target (review II.1).
    if "ess_retention" in inference_cfg:
        abcpmc_kwargs["ess_target"] = float(inference_cfg["ess_retention"])
    propagator = ABCPMC(**abcpmc_kwargs)

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
        propulator_kwargs = {
            "island_comm": propulate_comm,
            "propulate_comm": propulate_comm,
        }
        try:
            from mpi4py import MPI as _MPI_for_worker

            propulator_kwargs["worker_sub_comm"] = _MPI_for_worker.COMM_SELF
        except Exception:
            # No mpi4py available (unit-test path with a fake comm). The
            # propulator gets the fake comm in island/propulate slots; the
            # worker_sub_comm slot is left to its default and the drain loop
            # in _propulate_with_wall_time_limit no-ops when MPI is None.
            pass

    if os.environ.get("ASYNC_ABC_PHASE_TIMING"):
        propagator = _PhaseTimingPropagator(
            propagator, data_dir=output_dir.data, tag=_tag, replicate=replicate,
        )

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

    # In all-ranks execution mode, run_method_distributed keeps only ROOT's
    # records (runner.py: `return records if root_rank else []`); the post-run
    # sort + per-particle record build on every other rank is computed and then
    # discarded. At scaling volumes (~1e6 individuals/rank) that redundant build
    # is both 95x wasteful AND the dominant source of post-run rank desync: at 96
    # ranks across 2 nodes the slow ranks miss the teardown window and srun Force
    # Terminates the step (observed: only ~26/96 reach status=finish). Skip it on
    # non-root — the output is identical (root's records are the ones returned).
    # The barrier above already resynchronised all ranks, so non-root returning
    # here cannot desync the next replicate's Dup(). Flag set by
    # run_method_distributed only for all_ranks mode.
    if inference_cfg.get("_records_root_only") and not _comm_world_is_root():
        return []

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
    # Retroactive AMIS posterior weights. This is the estimator the paper's
    # consistency + CLT are stated for: every particle reweighted against the
    # cumulative proposal mixture, reconstructed from history (off the timed
    # inference path; see ABCPMC.extract_posterior). The streaming proposal-time
    # `ind.weight` is kept separately (records' `weight`) for the ESS-over-time
    # diagnostic. Computed on `population` in record order so the weights align
    # index-for-index; falls back to None on any error, in which case downstream
    # consumers (SBC) revert to the streaming weight.
    posterior_weights: List[Optional[float]] = [None] * len(population)
    if population and compute_posterior_weights:
        try:
            _, _retro = propagator.extract_posterior(population)
            if len(_retro) == len(population):
                posterior_weights = [float(w) for w in _retro]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "extract_posterior failed (%s); SBC will fall back to streaming weights.",
                exc,
            )
    elif population and not compute_posterior_weights:
        logger.info(
            "compute_posterior_weights=False: skipping retroactive AMIS reweighting "
            "for %d individuals (posterior_weight left empty; recompute offline if needed).",
            len(population),
        )

    records: List[ParticleRecord] = []
    current_tolerance = float(tol_init)
    for step, ind in enumerate(population, start=1):
        params = _individual_params(ind, limits)
        weight = float(ind.weight) if ind.weight is not None else None
        posterior_weight = posterior_weights[step - 1]
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
            posterior_weight=posterior_weight,
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
