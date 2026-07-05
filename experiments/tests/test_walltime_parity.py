"""W2.4 — uniform wall-time enforcement across inference methods.

Each method consumes ``max_wall_time_s`` and must respect it as a hard cap
(first-rank-hit semantics, not collective). This test pins a deliberately
slow simulator and asserts that each method exits within ``budget + epsilon``.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.inference._pyabc_common import Deadline
from async_abc.io.paths import OutputDir


# Per-call simulator sleep; the cumulative work easily exceeds the budget,
# so wall-time enforcement is the only path to a clean exit.
_SLEEP_S = 0.01
_BUDGET_S = 3.0
# Tolerance: pyABC's max_walltime is checked between populations (not per
# particle), so the cap can be exceeded by up to one population's duration.
# At k=10 with _SLEEP_S=0.01 a typical population is ~0.5-1.5s; 5s tolerance
# leaves comfortable headroom while still detecting unbounded runaway.
_TOLERANCE_S = 5.0


def _slow_gaussian_benchmark():
    """Gaussian-mean benchmark with a per-evaluation sleep."""
    bm = make_benchmark(
        {
            "name": "gaussian_mean",
            "observed_data_seed": 42,
            "n_obs": 20,
            "true_mu": 0.0,
            "sigma_obs": 1.0,
            "prior_low": -3.0,
            "prior_high": 3.0,
        }
    )
    real_simulate = bm.simulate

    def slow_simulate(params, seed):
        time.sleep(_SLEEP_S)
        return real_simulate(params, seed)

    bm.simulate = slow_simulate
    return bm


def _walltime_cfg(extra=None):
    cfg = {
        "k": 10,
        "tol_init": 1.0,
        "max_simulations": 100_000,
        "n_generations": 1000,
        "max_wall_time_s": _BUDGET_S,
        "scheduler_type": "acceptance_rate",
        "n_workers": 1,
    }
    if extra:
        cfg.update(extra)
    return cfg


def test_deadline_helper_uses_monotonic_clock():
    """The Deadline helper itself must respect the configured budget."""
    d = Deadline(max_wall_time_s=0.05)
    assert not d.expired
    time.sleep(0.07)
    assert d.expired
    assert d.remaining == 0.0


def test_deadline_helper_disabled_when_none():
    d = Deadline(max_wall_time_s=None)
    assert not d.expired
    assert d.remaining is None


def test_rejection_abc_respects_walltime(tmp_path):
    """rejection_abc polls Deadline between candidate draws."""
    from async_abc.inference.rejection_abc import run_rejection_abc

    bm = _slow_gaussian_benchmark()
    od = OutputDir(tmp_path, "rej").ensure()
    cfg = _walltime_cfg()

    t0 = time.monotonic()
    run_rejection_abc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)
    elapsed = time.monotonic() - t0

    assert elapsed <= _BUDGET_S + _TOLERANCE_S, (
        f"rejection_abc exceeded wall-time budget: {elapsed:.2f}s "
        f"vs budget {_BUDGET_S}s + tol {_TOLERANCE_S}s"
    )


def test_pyabc_smc_respects_walltime(tmp_path):
    pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
    from async_abc.inference.pyabc_wrapper import run_pyabc_smc

    bm = _slow_gaussian_benchmark()
    od = OutputDir(tmp_path, "pyabc").ensure()
    cfg = _walltime_cfg({"max_simulations": 5000, "kernel": "hard"})

    t0 = time.monotonic()
    run_pyabc_smc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)
    elapsed = time.monotonic() - t0

    assert elapsed <= _BUDGET_S + _TOLERANCE_S, (
        f"pyabc_smc exceeded wall-time budget: {elapsed:.2f}s "
        f"vs budget {_BUDGET_S}s + tol {_TOLERANCE_S}s"
    )


def test_abc_smc_baseline_respects_walltime(tmp_path):
    pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
    from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

    bm = _slow_gaussian_benchmark()
    od = OutputDir(tmp_path, "smc").ensure()
    cfg = _walltime_cfg({"n_generations": 2, "kernel": "hard"})

    t0 = time.monotonic()
    run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)
    elapsed = time.monotonic() - t0

    assert elapsed <= _BUDGET_S + _TOLERANCE_S, (
        f"abc_smc_baseline exceeded wall-time budget: {elapsed:.2f}s "
        f"vs budget {_BUDGET_S}s + tol {_TOLERANCE_S}s"
    )
