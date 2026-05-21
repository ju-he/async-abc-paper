"""Shared utilities for pyABC-based inference methods.

Contains helpers used by both :mod:`pyabc_wrapper` (pyabc_smc) and
:mod:`abc_smc_baseline` to avoid code duplication.

Includes :func:`make_acceptor`, the apples-to-apples bridge between the
propulate-side smooth-kernel ABC (hard / Gaussian / Epanechnikov) and pyABC's
``Acceptor`` protocol. The resulting acceptor uses the *same* kernel function
``K_eps(rho)`` as the propulate propagator, so the only methodological
difference between the propulate and pyABC runs in apples-to-apples mode is
the synchronisation regime.
"""
from pathlib import Path
from typing import Any

import numpy as np

from ..io.paths import OutputDir


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
        Seed for the local NumPy ``Generator`` used for the rejection step.
        Each replicate should pass a distinct seed.

    Returns
    -------
    pyabc.Acceptor
        Ready to pass to ``pyabc.ABCSMC(acceptor=...)``.
    """
    import pyabc
    from pyabc.acceptor import Acceptor, AcceptorResult
    from propulate.propagators.abcpmc import _make_kernel

    if kernel == "hard":
        return pyabc.UniformAcceptor()

    kfn = _make_kernel(kernel)
    rng = np.random.default_rng(rng_seed)

    class _SmoothKernelAcceptor(Acceptor):
        """Probabilistic-rejection smooth-kernel ABC acceptor.

        Implements the canonical smooth-ABC scheme (Wilkinson 2013): a
        candidate with discrepancy ``rho`` and bandwidth ``eps`` is accepted
        with probability ``K_eps(rho)`` (peak-normalised so ``K_eps(0) = 1``).
        Accepted particles enter pyABC's importance-sampling machinery with
        weight 1, matching pyABC's standard SMC bookkeeping.
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
            accept = bool(rng.random() < p_accept)
            return AcceptorResult(distance=d, accept=accept, weight=1.0)

    return _SmoothKernelAcceptor()
