"""Shared utilities for pyABC-based inference methods.

Contains helpers used by both :mod:`pyabc_wrapper` (pyabc_smc) and
:mod:`abc_smc_baseline` to avoid code duplication.

Includes:

- :func:`make_acceptor`, the apples-to-apples bridge between the
  propulate-side smooth-kernel ABC (hard / Gaussian / Epanechnikov) and
  pyABC's ``Acceptor`` protocol. The resulting acceptor uses the *same*
  kernel function ``K_eps(rho)`` as the propulate propagator, so the only
  methodological difference between the propulate and pyABC runs in
  apples-to-apples mode is the synchronisation regime.
- :class:`Deadline`, a monotonic-clock deadline helper used by every
  wrapper to enforce ``max_wall_time_s`` uniformly. The deadline reports
  *first-rank-hit* semantics — once any rank trips the deadline it raises,
  and the wrapper must serialise its current state to the records before
  returning.
"""
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..io.paths import OutputDir
from ..utils.mpi import get_rank
from ..utils.seeding import stable_seed

# Per-process acceptance-RNG streams, keyed by (replicate seed, MPI rank,
# pyABC generation index). pyABC's MappingSampler cloudpickles the acceptor
# to the workers and unpickles it freshly FOR EVERY WORK ITEM
# (pyabc/sampler/mapping.py::map_function), so instance state cannot carry a
# random stream: it would restart identically on every item, and every
# worker would share the root copy's initial state (cross-worker correlated
# acceptance decisions). A module-level cache gives each (seed, rank, t) an
# independent, reproducible stream that keeps advancing across items within
# a generation.
_ACCEPTOR_STREAMS: Dict[Tuple[int, int, int], "np.random.Generator"] = {}


def _acceptor_rng(rng_seed: int, t: int) -> "np.random.Generator":
    """Return the per-(seed, rank, generation) acceptance stream."""
    key = (int(rng_seed), get_rank(), int(t))
    rng = _ACCEPTOR_STREAMS.get(key)
    if rng is None:
        rng = np.random.default_rng(stable_seed("acceptor", *key))
        _ACCEPTOR_STREAMS[key] = rng
    return rng


def _reset_acceptor_streams() -> None:
    """Test hook: forget all cached acceptance streams."""
    _ACCEPTOR_STREAMS.clear()


class Deadline:
    """Monotonic-clock wall-time deadline shared by all inference wrappers.

    Construct with ``max_wall_time_s`` (None disables enforcement). Each
    wrapper consults ``expired`` between expensive units of work
    (per-particle for rejection ABC, per-population for pyABC). The class
    uses ``time.monotonic`` so the deadline is unaffected by system-clock
    adjustments.

    Use ``configure_pyabc_max_walltime()`` to also push the same deadline
    into pyABC's internal ``max_walltime`` mechanism for double safety.
    """

    __slots__ = ("_start", "_budget_s")

    def __init__(self, max_wall_time_s: Optional[float]) -> None:
        self._start: float = time.monotonic()
        self._budget_s: Optional[float] = (
            None if max_wall_time_s is None else float(max_wall_time_s)
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def budget(self) -> Optional[float]:
        return self._budget_s

    @property
    def expired(self) -> bool:
        if self._budget_s is None:
            return False
        return self.elapsed >= self._budget_s

    @property
    def remaining(self) -> Optional[float]:
        if self._budget_s is None:
            return None
        return max(0.0, self._budget_s - self.elapsed)


def db_suffix(checkpoint_tag: str) -> str:
    """Return a filesystem-safe suffix derived from *checkpoint_tag*."""
    if not checkpoint_tag:
        return ""
    safe_tag = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(checkpoint_tag)
    )
    return f"__{safe_tag}" if safe_tag else ""


def prepare_db_path(
    output_dir: OutputDir,
    *,
    method_name: str,
    replicate: int,
    seed: int,
    checkpoint_tag: str,
) -> str:
    """Create (or clean) a SQLite database path for a pyABC run."""
    db_file = (
        output_dir.data
        / f"{method_name}_rep{replicate}_seed{seed}{db_suffix(checkpoint_tag)}.db"
    )
    for path in (db_file, Path(f"{db_file}-wal"), Path(f"{db_file}-shm")):
        if path.exists():
            path.unlink()
    return f"sqlite:///{db_file}"


def make_acceptor(kernel: str, rng_seed: int) -> Any:
    """Build a pyABC ``Acceptor`` matching the propulate-side ABC kernel.

    For ``kernel="hard"`` returns pyABC's default ``UniformAcceptor`` so the
    legacy behaviour is preserved bit-for-bit. For ``"gaussian"`` and
    ``"epanechnikov"`` returns a probabilistic-rejection acceptor:
    a candidate with discrepancy ``rho`` is accepted with probability
    ``K_eps(rho) / K_eps(0)`` (== ``K_eps(rho)`` for the normalised forms
    used here), giving the same effective smooth-kernel ABC likelihood as
    the propulate propagator. This makes the pyABC and propulate baselines
    apples-to-apples on the kernel: they share ``K_eps(rho)``, differ only
    on the synchronisation regime (generation barrier vs. steady-state).

    Parameters
    ----------
    kernel:
        ``"hard"`` | ``"gaussian"`` | ``"epanechnikov"``.
    rng_seed:
        Base seed for the rejection-step streams. Each replicate should pass
        a distinct seed. The actual draw comes from a per-process stream
        keyed by ``(rng_seed, MPI rank, generation)`` (BLAKE2b-derived via
        ``stable_seed``), so acceptance decisions are decorrelated across
        workers and reproducible from the replicate seed — see
        :func:`_acceptor_rng` for why instance state cannot hold the stream.

    Returns
    -------
    pyabc.Acceptor
        Ready to pass to ``pyabc.ABCSMC(acceptor=...)``.
    """
    import pyabc

    if kernel == "hard":
        return pyabc.UniformAcceptor()

    from pyabc.acceptor import Acceptor, AcceptorResult
    from propulate.propagators.abcpmc import _make_kernel

    kfn = _make_kernel(kernel)
    rng_seed = int(rng_seed)

    class _SmoothKernelAcceptor(Acceptor):
        """Probabilistic-rejection smooth-kernel ABC acceptor.

        Implements the canonical smooth-ABC scheme (Wilkinson 2013): a
        candidate with discrepancy ``rho`` and bandwidth ``eps`` is accepted
        with probability ``K_eps(rho)`` (peak-normalised so ``K_eps(0) = 1``).
        Accepted particles enter pyABC's importance-sampling machinery with
        weight 1, matching pyABC's standard SMC bookkeeping.

        The rejection draw uses the module-level per-(seed, rank, generation)
        stream (:func:`_acceptor_rng`), NOT instance state: MappingSampler
        unpickles this object anew for every work item, so an instance-held
        Generator would restart identically per item and be shared (same
        initial state) across all workers.
        """

        def __init__(self) -> None:
            super().__init__()
            self.kernel_name = kernel

        def __call__(
            self,
            distance_function,
            eps,
            x,
            x_0,
            t,
            par,
        ):
            d = float(distance_function(x, x_0, t, par))
            eps_t = float(eps(t))
            w_arr = kfn.weight(np.array([d]), eps_t)
            w = float(w_arr[0])
            # K_eps is normalised so K_eps(0) = 1 for hard/gaussian/epanechnikov;
            # acceptance probability is therefore K_eps(rho) directly. Clamp to
            # [0, 1] defensively in case of numerical edge cases.
            p_accept = min(max(w, 0.0), 1.0)
            accept = bool(_acceptor_rng(rng_seed, t).random() < p_accept)
            return AcceptorResult(distance=d, accept=accept, weight=1.0)

    return _SmoothKernelAcceptor()


def make_matched_epsilon(
    kernel: str,
    tol_init: float,
    *,
    ess_retention: float = 0.95,
    max_tighten_factor: float = 0.5,
) -> Any:
    """Build a pyABC ``Epsilon`` applying the propulate ε rule per generation.

    The asynchronous side selects its bandwidth per arrival with the
    kernel-weighted ESS-retention rule (``select_eps_by_ess_retention`` in
    ``propulate.propagators.abcpmc``). This adapter applies the *identical*
    implementation once per pyABC generation — the granularity a
    generation-staged sampler permits — on the finished population's weighted
    distances, clamped monotone. Kernel and ε rule are thus both shared
    between the two arms of the comparison; the remaining difference is the
    update granularity (per arrival vs per generation), which is exactly the
    synchronization variable under test.

    Parameters
    ----------
    kernel:
        ``"gaussian"`` | ``"epanechnikov"``. The hard kernel is rejected:
        its ESS is a step function in ε, so the retention rule is degenerate
        — use ``pyabc.QuantileEpsilon`` for hard-kernel runs.
    tol_init:
        Initial tolerance ε₀, same value the asynchronous side starts from.
    ess_retention:
        Retention target α (async side's ``ess_target``), default 0.95.
    max_tighten_factor:
        Per-generation tightening floor, default 0.5 (async side's default).

    Returns
    -------
    pyabc.Epsilon
        Ready to pass to ``pyabc.ABCSMC(eps=...)``.
    """
    import pyabc  # noqa: F401 -- clear ImportError if the dependency is missing
    from pyabc.epsilon import QuantileEpsilon

    from propulate.propagators.abcpmc import (
        _make_kernel,
        select_eps_by_ess_retention,
    )

    if kernel == "hard":
        raise ValueError(
            "make_matched_epsilon requires a smooth kernel; the hard kernel's "
            "ESS is a step function in eps. Use pyabc.QuantileEpsilon "
            "(epsilon_mode='quantile') for kernel='hard'."
        )
    kfn = _make_kernel(kernel)
    tol_init = float(tol_init)

    class _MatchedKernelEpsilon(QuantileEpsilon):
        """ESS-retention epsilon matched to the asynchronous side.

        Reuses QuantileEpsilon's ``initialize``/``__call__``/``update``
        lookup plumbing (numeric initial epsilon, so no calibration sample);
        only the ε computation (``_update``) differs.
        """

        def __init__(self) -> None:
            super().__init__(initial_epsilon=tol_init, alpha=0.5)
            self.kernel_name = kernel
            self.ess_retention = float(ess_retention)
            self.max_tighten_factor = float(max_tighten_factor)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "rule": "ess_retention",
                    "kernel": self.kernel_name,
                    "ess_retention": self.ess_retention,
                    "max_tighten_factor": self.max_tighten_factor,
                }
            )
            return config

        def _update(self, t: int, weighted_distances) -> None:
            # pyABC calls update(t) after generation t-1 finishes, asking for
            # generation t's epsilon; the finished generation's epsilon is the
            # current tolerance the retention rule tightens from. A missing
            # t-1 entry means the call convention changed — crash loudly.
            eps_prev = float(self._look_up[t - 1])
            distances = weighted_distances.distance.values.astype(float)
            weights = weighted_distances.w.values.astype(float)
            proposal = select_eps_by_ess_retention(
                weights,
                distances,
                eps_prev,
                kfn,
                ess_target=self.ess_retention,
                max_tighten_factor=self.max_tighten_factor,
            )
            # Monotone clamp, mirroring the async side's min(ε_hist, ε_sched).
            self._look_up[t] = min(eps_prev, float(proposal))

    return _MatchedKernelEpsilon()
