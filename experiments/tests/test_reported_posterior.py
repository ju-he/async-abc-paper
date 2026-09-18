"""The reported-posterior replay must reproduce the run that produced it.

These tests exist because the paper's inference claims are about
``ABCPMC.extract_posterior`` -- the retroactive AMIS posterior -- and every
figure that scores it post hoc rebuilds it from ``raw_results.csv``. If that
rebuild is not faithful, the figures are measuring something else.
"""
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "propulate"))

from async_abc.analysis.reported_posterior import (  # noqa: E402
    individuals_from_records,
    infer_n_bootstrap,
    reported_posterior,
    reported_posterior_curve,
    weighted_w1,
)
from async_abc.io.records import ParticleRecord  # noqa: E402

LIMITS = {"mu": (-5.0, 5.0)}


def _drive(n=900, k=100, seed=11):
    """Run the shipped propagator serially; it is a pure function of history."""
    from propulate.propagators.abcpmc import ABCPMC

    rng = np.random.default_rng(seed)
    ybar = float(rng.normal(0.0, 0.1))
    prop = ABCPMC(LIMITS, k=k, kernel="gaussian", scheduler_type="acceptance_rate",
                  amis_snapshots=20, perturbation_scale=0.8, tol=5.0,
                  rng=random.Random(seed))
    hist = []
    for i in range(n):
        child = prop(hist)
        child.generation = i
        child.loss = abs(float(rng.normal(child.position[0], 0.1)) - ybar)
        hist.append(child)
    return prop, hist, ybar


def _records(hist):
    """Serialise exactly as the runner does: running-min `tolerance` for the
    plots, verbatim stamped value in `proposal_tolerance` for the replay."""
    out, running = [], 5.0
    for i, h in enumerate(hist):
        if h.tolerance is not None:
            running = min(running, float(h.tolerance))
        out.append(ParticleRecord(
            method="async_propulate_abc", replicate=0, seed=0, step=i + 1,
            params={"mu": float(h.position[0])}, loss=h.loss, weight=h.weight,
            tolerance=running,
            proposal_tolerance=None if h.tolerance is None else float(h.tolerance),
            wall_time=float(i),
        ))
    return out


class TestReplayFidelity:
    def test_replay_from_records_is_bit_identical_to_the_live_run(self):
        prop, hist, _ = _drive()
        _, w_live = prop.extract_posterior(hist)
        _, w_replay = reported_posterior(
            _records(hist), LIMITS, k=100, kernel="gaussian",
            scheduler_type="acceptance_rate", amis_snapshots=20,
            perturbation_scale=0.8, tol=5.0,
        )
        assert np.allclose(np.asarray(w_live, float), w_replay, rtol=0, atol=1e-12)

    def test_losing_the_bootstrap_marker_changes_the_answer(self):
        """Guards the reason `proposal_tolerance` exists. If this ever stops
        failing, the extra column is dead weight and can go."""
        prop, hist, _ = _drive()
        _, w_live = np.asarray(prop.extract_posterior(hist)[0]), np.asarray(
            prop.extract_posterior(hist)[1], float)
        recs = _records(hist)
        for r in recs:               # prior phase collapsed into tol_init
            r.proposal_tolerance = None if r.tolerance is None else r.tolerance
        _, w_lossy = reported_posterior(
            recs, LIMITS, k=100, kernel="gaussian",
            scheduler_type="acceptance_rate", amis_snapshots=20,
            perturbation_scale=0.8, tol=5.0, n_bootstrap=0,
        )
        assert not np.allclose(w_live, w_lossy, rtol=0, atol=1e-9)

    def test_infer_n_bootstrap_reads_the_boundary_off_weight(self):
        """Prior draws have weight exactly 1.0 (proposal == prior) and form a
        contiguous leading run, so the boundary needs no fitting."""
        _, hist, _ = _drive()
        truth = sum(1 for h in hist if h.tolerance is None)
        assert truth > 0
        assert infer_n_bootstrap(_records(hist)) == truth

    def test_legacy_records_replay_exactly_without_the_field(self):
        """A file written before proposal_tolerance existed still reproduces its
        own reported posterior, because the boundary is inferred rather than
        guessed."""
        prop, hist, _ = _drive()
        _, w_live = prop.extract_posterior(hist)
        recs = _records(hist)
        for r in recs:               # exactly the pre-fix serialisation
            r.proposal_tolerance = None
            r.tolerance = r.tolerance if r.tolerance is not None else 5.0
        _, w = reported_posterior(
            recs, LIMITS, k=100, kernel="gaussian", scheduler_type="acceptance_rate",
            amis_snapshots=20, perturbation_scale=0.8, tol=5.0,
        )
        assert np.allclose(np.asarray(w_live, float), w, rtol=0, atol=1e-12)


class TestMetricHasResolution:
    """The point mass metric cannot tell a correct posterior from a collapsed
    one; W1 against a reference posterior can. This is the whole reason the
    recovery figures are being rebuilt."""

    def test_point_mass_metric_prefers_a_collapsed_posterior(self):
        from async_abc.analysis.convergence import _wasserstein_to_true_params
        import pandas as pd

        rng = np.random.default_rng(0)
        truth = {"mu": 0.0}
        correct = pd.DataFrame({"mu": rng.normal(0.0, 0.1, 20000)})
        collapsed = pd.DataFrame({"mu": rng.normal(0.0, 0.001, 20000)})
        assert (_wasserstein_to_true_params(collapsed, truth, 50)
                < _wasserstein_to_true_params(correct, truth, 50))

    def test_reference_metric_prefers_the_correct_posterior(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0.0, 0.1, 20000).reshape(-1, 1)
        correct = rng.normal(0.0, 0.1, 20000).reshape(-1, 1)
        collapsed = rng.normal(0.0, 0.001, 20000).reshape(-1, 1)
        assert (weighted_w1(correct, None, reference)
                < weighted_w1(collapsed, None, reference))

    def test_point_mass_floor_is_the_posterior_spread(self):
        """An exactly correct posterior scores its own mean absolute deviation,
        not zero -- which is why every arm of the straggler twin scored 0.07-0.08."""
        from async_abc.analysis.convergence import _wasserstein_to_true_params
        import pandas as pd

        rng = np.random.default_rng(0)
        sd = 0.1
        correct = pd.DataFrame({"mu": rng.normal(0.0, sd, 200000)})
        assert _wasserstein_to_true_params(correct, {"mu": 0.0}, 50) == pytest.approx(
            sd * np.sqrt(2 / np.pi), rel=0.02
        )


class TestCurve:
    def test_curve_tracks_prefixes_and_improves(self):
        prop, hist, ybar = _drive(n=1200)
        reference = np.random.default_rng(1).normal(ybar, 0.1, 20000).reshape(-1, 1)
        df = reported_posterior_curve(
            _records(hist), LIMITS, reference, checkpoints=5, min_records=200,
            k=100, kernel="gaussian", scheduler_type="acceptance_rate",
            amis_snapshots=20, perturbation_scale=0.8, tol=5.0,
        )
        assert len(df) == 5
        assert df["n_records"].is_monotonic_increasing
        assert df["wall_time"].is_monotonic_increasing
        assert df["w1"].iloc[-1] < df["w1"].iloc[0]
        assert ((df["ess_fraction"] > 0) & (df["ess_fraction"] <= 1)).all()

    def test_curve_is_empty_below_the_minimum(self):
        _, hist, _ = _drive(n=120)
        df = reported_posterior_curve(
            _records(hist), LIMITS, np.zeros((10, 1)), min_records=500,
        )
        assert df.empty


class TestEffectiveSampleSize:
    """The reported estimator's ESS is set by the archive size, not the budget.

    Measured on the production Gaussian run: ESS 261-303 out of 1.1-1.3M
    evaluated particles. ``scripts/diag_ess_growth.py`` is the controlled
    version; this is the regression guard, since a change that made ESS grow
    with n would be a change to what the paper reports.
    """

    @staticmethod
    def _ess(k, n, seed=17):
        from propulate.propagators.abcpmc import ABCPMC

        rng = np.random.default_rng(seed)
        ybar = float(rng.normal(0.0, 0.1))
        prop = ABCPMC(LIMITS, k=k, kernel="gaussian",
                      scheduler_type="acceptance_rate", amis_snapshots=20,
                      perturbation_scale=0.8, tol=5.0, rng=random.Random(seed))
        hist = []
        for i in range(n):
            c = prop(hist)
            c.generation = i
            c.loss = abs(float(rng.normal(c.position[0], 0.1)) - ybar)
            hist.append(c)
        _, w = prop.extract_posterior(hist)
        w = np.asarray(w, float)
        w = w / w.sum()
        return float(1.0 / np.sum(w ** 2))

    def test_ess_does_not_grow_with_the_budget(self):
        small, large = self._ess(50, 2_000), self._ess(50, 8_000)
        assert large < 3 * small, (
            f"ESS grew from {small:.0f} to {large:.0f} over a 4x budget; if this "
            "is a real improvement the paper's flat accuracy curve and its "
            "effective-support discussion both need revisiting"
        )

    def test_ess_scales_with_the_archive_size(self):
        assert self._ess(100, 4_000) > 1.5 * self._ess(30, 4_000)
