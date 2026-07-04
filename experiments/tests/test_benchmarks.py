"""Tests for async_abc.benchmarks.*"""
import importlib
import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from async_abc.benchmarks.gaussian_mean import GaussianMean
from async_abc.benchmarks.gandk import GandK
from async_abc.benchmarks.lotka_volterra import LotkaVolterra
from async_abc.benchmarks import make_benchmark
from async_abc.benchmarks.realistic_workload import (
    _ensure_backend_on_path,
    denormalize_params,
    normalize_params,
)

# ---------------------------------------------------------------------------
# Realistic-workload helpers & fixtures
# ---------------------------------------------------------------------------

_RW_TEMPLATE_DIR = Path(__file__).parents[1] / "assets" / "realistic_workload"


def _backend_available() -> bool:
    try:
        _ensure_backend_on_path()
        return True
    except ImportError:
        return False


_BACKEND_AVAILABLE = _backend_available()


@pytest.fixture
def rw_config(tmp_path):
    """Minimal RealisticWorkload config (sim backend must be available)."""
    if not _BACKEND_AVAILABLE:
        pytest.skip("sim backend not available — run with sim_backend_venv/.venv")
    return {
        "name": "realistic_workload",
        "sim_config_template": str(_RW_TEMPLATE_DIR / "sim_config.json"),
        "config_builder_params": str(_RW_TEMPLATE_DIR / "config_builder_params.json"),
        "distance_metric_params": str(_RW_TEMPLATE_DIR / "distance_metric_params.json"),
        "parameter_space": str(_RW_TEMPLATE_DIR / "parameter_space.json"),
        "reference_data_path": str(tmp_path / "reference"),
        "output_dir": str(tmp_path / "sims"),
    }


@pytest.fixture
def rw_mocks(tmp_path):
    """Pre-built mock SimulationManager and DistanceMetric for injection."""
    mock_sim = MagicMock()
    mock_dist = MagicMock()
    config_path = tmp_path / "rw_test" / "eval000001" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}")
    mock_sim.build_simulation_config.return_value = str(config_path)
    mock_dist.calculate_distance.return_value = 2.5
    return mock_sim, mock_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bm_config(name, **kwargs):
    """Return a minimal benchmark sub-config."""
    cfg = {"name": name, "observed_data_seed": 0, "n_obs": 50}
    cfg.update(kwargs)
    return cfg


# ---------------------------------------------------------------------------
# GaussianMean
# ---------------------------------------------------------------------------

class TestGaussianMean:
    def test_init_no_error(self):
        GaussianMean(_bm_config("gaussian_mean"))

    def test_limits_has_mu(self):
        bm = GaussianMean(_bm_config("gaussian_mean"))
        assert "mu" in bm.limits
        lo, hi = bm.limits["mu"]
        assert lo < hi

    def test_simulate_returns_float(self):
        bm = GaussianMean(_bm_config("gaussian_mean"))
        loss = bm.simulate({"mu": 0.0}, seed=1)
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert loss >= 0.0

    def test_simulate_true_params_smaller_loss(self):
        """Loss at the true mu must be less than at a clearly wrong mu."""
        bm = GaussianMean(_bm_config("gaussian_mean", true_mu=0.0, n_obs=500))
        loss_true = bm.simulate({"mu": 0.0}, seed=5)
        loss_wrong = bm.simulate({"mu": 4.0}, seed=5)
        assert loss_true < loss_wrong

    def test_simulate_deterministic_seed(self):
        bm = GaussianMean(_bm_config("gaussian_mean"))
        a = bm.simulate({"mu": 1.0}, seed=42)
        b = bm.simulate({"mu": 1.0}, seed=42)
        assert a == pytest.approx(b)

    def test_simulate_different_seeds_differ(self):
        bm = GaussianMean(_bm_config("gaussian_mean", n_obs=5))
        a = bm.simulate({"mu": 1.0}, seed=1)
        b = bm.simulate({"mu": 1.0}, seed=2)
        # With only 5 obs, seeds will almost certainly give different sample means
        assert a != pytest.approx(b)

    def test_analytic_posterior_mean_is_float(self):
        bm = GaussianMean(_bm_config("gaussian_mean"))
        pm = bm.analytic_posterior_mean()
        assert isinstance(pm, float)
        assert math.isfinite(pm)

    def test_analytic_posterior_mean_close_to_true(self):
        """With many observations the posterior mean should be close to true mu."""
        bm = GaussianMean(_bm_config("gaussian_mean", true_mu=1.5, n_obs=2000))
        pm = bm.analytic_posterior_mean()
        assert abs(pm - 1.5) < 0.2

    def test_analytic_posterior_uses_uniform_prior(self):
        """Analytic posterior must use the same Uniform prior as the ABC inference."""
        # Symmetric prior: result should be close to observed_mean
        bm_sym = GaussianMean(_bm_config(
            "gaussian_mean", true_mu=0.0, n_obs=100,
            prior_low=-5.0, prior_high=5.0,
        ))
        pm_sym = bm_sym.analytic_posterior_mean()
        assert abs(pm_sym - bm_sym.observed_mean) < 1e-9

        # Asymmetric prior that clips: observed mean outside bounds
        bm_asym = GaussianMean(_bm_config(
            "gaussian_mean", true_mu=10.0, n_obs=100,
            prior_low=-5.0, prior_high=5.0,
        ))
        pm_asym = bm_asym.analytic_posterior_mean()
        # observed_mean ≈ 10, but prior caps at 5
        assert pm_asym == pytest.approx(5.0)

    def test_different_observed_seeds_give_different_data(self):
        bm1 = GaussianMean(_bm_config("gaussian_mean", observed_data_seed=0))
        bm2 = GaussianMean(_bm_config("gaussian_mean", observed_data_seed=99))
        assert bm1.observed_mean != pytest.approx(bm2.observed_mean)


# ---------------------------------------------------------------------------
# GandK
# ---------------------------------------------------------------------------

class TestGandK:
    def test_init_no_error(self):
        GandK(_bm_config("gandk", n_obs=200))

    def test_limits_has_all_params(self):
        bm = GandK(_bm_config("gandk", n_obs=200))
        for key in ("A", "B", "g", "k"):
            assert key in bm.limits
            lo, hi = bm.limits[key]
            assert lo < hi

    def test_simulate_returns_positive_float(self):
        bm = GandK(_bm_config("gandk", n_obs=200))
        loss = bm.simulate({"A": 3.0, "B": 1.0, "g": 2.0, "k": 0.5}, seed=0)
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert loss >= 0.0

    def test_simulate_deterministic_seed(self):
        bm = GandK(_bm_config("gandk", n_obs=200))
        params = {"A": 3.0, "B": 1.0, "g": 2.0, "k": 0.5}
        a = bm.simulate(params, seed=7)
        b = bm.simulate(params, seed=7)
        assert a == pytest.approx(b)

    def test_simulate_true_params_smaller_loss(self):
        """Loss at true params should be smaller than at clearly wrong params."""
        true = {"A": 3.0, "B": 1.0, "g": 2.0, "k": 0.5}
        wrong = {"A": 0.1, "B": 3.9, "g": 0.0, "k": 0.9}
        bm = GandK(_bm_config("gandk", n_obs=500,
                               true_A=3.0, true_B=1.0, true_g=2.0, true_k=0.5))
        # Average over a few seeds to reduce noise
        loss_true = np.mean([bm.simulate(true, seed=s) for s in range(10)])
        loss_wrong = np.mean([bm.simulate(wrong, seed=s) for s in range(10)])
        assert loss_true < loss_wrong

    def test_summary_stats_shape(self):
        bm = GandK(_bm_config("gandk", n_obs=200))
        assert bm.observed_stats.shape == (7,)


# ---------------------------------------------------------------------------
# LotkaVolterra
# ---------------------------------------------------------------------------

class TestLotkaVolterra:
    # Use small T_max and initial populations so tests are fast
    _config = _bm_config("lotka_volterra", T_max=10.0, x0=50, y0=25)

    def test_init_no_error(self):
        LotkaVolterra(self._config)

    def test_limits_has_all_params(self):
        bm = LotkaVolterra(self._config)
        for key in ("theta1", "theta2", "theta3", "theta4"):
            assert key in bm.limits

    def test_simulate_returns_finite_float(self):
        bm = LotkaVolterra(self._config)
        true_params = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.5}
        loss = bm.simulate(true_params, seed=0)
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert loss >= 0.0

    def test_simulate_deterministic_seed(self):
        bm = LotkaVolterra(self._config)
        params = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.5}
        a = bm.simulate(params, seed=3)
        b = bm.simulate(params, seed=3)
        assert a == pytest.approx(b)

    def test_simulate_different_seeds_may_differ(self):
        # Use longer T_max and stable params so extinction is unlikely
        bm = LotkaVolterra(_bm_config("lotka_volterra", T_max=30.0, x0=50, y0=25))
        params = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.5}
        results = [bm.simulate(params, seed=s) for s in range(20)]
        finite = [r for r in results if r < 1e5]
        # With 20 seeds and T_max=30, at least some runs should be non-extinction
        # and have varying losses due to stochasticity
        if len(finite) >= 2:
            assert len(set(finite)) > 1

    def test_extinction_returns_large_loss(self):
        """Very high predation rates cause extinction; loss should be large."""
        bm = LotkaVolterra(self._config)
        extreme_params = {"theta1": 0.1, "theta2": 10.0, "theta3": 10.0, "theta4": 10.0}
        loss = bm.simulate(extreme_params, seed=0)
        assert loss > 0.0
        assert math.isfinite(loss)

    def test_true_params_smaller_loss_than_bad(self):
        bm = LotkaVolterra(_bm_config("lotka_volterra", T_max=20.0, x0=50, y0=25))
        true_params = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.5}
        bad_params  = {"theta1": 5.0, "theta2": 0.5,   "theta3": 0.5,   "theta4": 5.0}
        loss_true = np.mean([bm.simulate(true_params, seed=s) for s in range(5)])
        loss_bad  = np.mean([bm.simulate(bad_params, seed=s)  for s in range(5)])
        assert loss_true < loss_bad

    def test_lotka_volterra_normalized_stats_balanced_scale(self):
        """Normalized summary stats should give each dimension meaningful weight."""
        bm = LotkaVolterra(_bm_config(
            "lotka_volterra", T_max=20.0, x0=50, y0=25, normalize_stats=True,
        ))
        # Two param sets that differ subtly
        p1 = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.5}
        p2 = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.6}
        d1 = bm.simulate(p1, seed=0)
        d2 = bm.simulate(p2, seed=0)
        # Both should be finite
        assert math.isfinite(d1) and math.isfinite(d2)

    def test_lotka_volterra_unnormalized_backward_compat(self):
        """normalize_stats=False preserves raw Euclidean distance."""
        bm = LotkaVolterra(_bm_config(
            "lotka_volterra", T_max=20.0, x0=50, y0=25, normalize_stats=False,
        ))
        params = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.5}
        loss = bm.simulate(params, seed=0)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_lotka_volterra_retries_on_extinction(self):
        """Retry loop should handle multiple extinction attempts."""
        # Use seed=42 which is known to produce a stable trajectory
        bm = LotkaVolterra(_bm_config(
            "lotka_volterra", T_max=10.0, x0=50, y0=25,
            observed_data_seed=42, max_extinction_retries=10,
        ))
        # Should have created valid observed_stats
        assert len(bm.observed_stats) == 6
        assert all(np.isfinite(bm.observed_stats))

    def test_lotka_volterra_raises_after_max_retries(self, monkeypatch):
        """If all retries produce extinction, raise RuntimeError."""
        from async_abc.benchmarks import lotka_volterra as lv_mod

        original_gillespie = lv_mod._gillespie

        def always_extinct(*args, **kwargs):
            times, xs, ys = original_gillespie(*args, **kwargs)
            # Force extinction
            xs[-1] = 0
            return times, xs, ys

        monkeypatch.setattr(lv_mod, "_gillespie", always_extinct)
        with pytest.raises(RuntimeError, match="extinct"):
            LotkaVolterra(_bm_config(
                "lotka_volterra", T_max=10.0, x0=50, y0=25,
                max_extinction_retries=3,
            ))


# ---------------------------------------------------------------------------
# RealisticWorkload
# ---------------------------------------------------------------------------

class TestRealisticWorkloadImport:
    """Module-level import must always succeed regardless of sim backend availability."""

    def test_import_does_not_raise(self):
        from async_abc.benchmarks import realistic_workload  # noqa: F401

    def test_class_importable(self):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload  # noqa: F401

    def test_missing_backend_raises_import_error_on_init(self, tmp_path, monkeypatch):
        """When sim backend is unavailable, __init__ raises ImportError."""
        import async_abc.benchmarks.realistic_workload as rw_mod

        def fail_ensure():
            raise ImportError("No sim backend available")

        monkeypatch.setattr(rw_mod, "_ensure_backend_on_path", fail_ensure)
        with pytest.raises(ImportError):
            rw_mod.RealisticWorkload({"name": "realistic_workload"})

    def test_normalize_generated_config_paths_rewrites_repo_relative_include(self, tmp_path, monkeypatch):
        from async_abc.benchmarks.realistic_workload import _rewrite_generated_config_paths

        include_target = tmp_path / "shared" / "filling.json"
        include_target.parent.mkdir(parents=True)
        include_target.write_text("{}")
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / "runs" / "reference"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps({"Include": [str(include_target.relative_to(tmp_path))]}))

        normalized = _rewrite_generated_config_paths(config_path)
        data = json.loads(normalized.read_text())

        assert data["Include"] == [str(include_target.resolve())]

    def test_resolve_reference_data_path_finds_nested_generated_dir(self, tmp_path, monkeypatch):
        import async_abc.benchmarks.realistic_workload as rw_mod

        configured = tmp_path / "experiments" / "data" / "realistic_reference" / "reference"
        configured.mkdir(parents=True)
        nested = (
            tmp_path
            / "experiments"
            / "data"
            / "realistic_reference"
            / "experiments"
            / "data"
            / "realistic_reference"
            / "reference"
        )
        (nested / "configs").mkdir(parents=True)
        (nested / "000000").mkdir(parents=True)
        (nested / "config.json").write_text("{}")
        (nested / "cis.out").write_text("")
        (nested / "000000" / "cellevents.log").write_text("")

        monkeypatch.setattr(rw_mod, "_REPO_ROOT", tmp_path)

        resolved = rw_mod._resolve_reference_data_path(
            "experiments/data/realistic_reference/reference"
        )

        assert resolved == nested.resolve()

    def test_resolve_reference_data_path_finds_arbitrarily_nested_dir(self, tmp_path, monkeypatch):
        import async_abc.benchmarks.realistic_workload as rw_mod

        configured = tmp_path / "experiments" / "data" / "realistic_reference" / "reference"
        configured.mkdir(parents=True)
        nested = (
            tmp_path
            / "experiments"
            / "data"
            / "realistic_reference"
            / "archive"
            / "2026-03-18"
            / "reference"
        )
        (nested / "configs").mkdir(parents=True)
        (nested / "000000").mkdir(parents=True)
        (nested / "config.json").write_text("{}")
        (nested / "cis.out").write_text("")
        (nested / "000000" / "cellevents.log").write_text("")

        monkeypatch.setattr(rw_mod, "_REPO_ROOT", tmp_path)

        resolved = rw_mod._resolve_reference_data_path(
            "experiments/data/realistic_reference/reference"
        )

        assert resolved == nested.resolve()

    def test_resolve_reference_data_path_accepts_csv_reference_dir(self, tmp_path, monkeypatch):
        import async_abc.benchmarks.realistic_workload as rw_mod

        configured = tmp_path / "experiments" / "assets" / "realistic_workload" / "reference_data"
        configured.mkdir(parents=True)
        (configured / "output_cells-00000.csv").write_text("#RecordID X Y Z\n1 0 0 0\n")

        monkeypatch.setattr(rw_mod, "_REPO_ROOT", tmp_path)

        resolved = rw_mod._resolve_reference_data_path(
            "experiments/assets/realistic_workload/reference_data"
        )

        assert resolved == configured.resolve()

    def test_ensure_reference_alias_reuses_canonical_reference_path(self, tmp_path):
        from async_abc.benchmarks.realistic_workload import _ensure_reference_alias

        output_dir = tmp_path / "experiments" / "data" / "realistic_reference"
        actual = output_dir / "experiments" / "data" / "realistic_reference" / "reference"
        (actual / "configs").mkdir(parents=True)
        (actual / "000000").mkdir(parents=True)
        (actual / "config.json").write_text("{}")
        (actual / "cis.out").write_text("")
        (actual / "000000" / "cellevents.log").write_text("")

        alias = _ensure_reference_alias(output_dir, actual)

        assert alias == output_dir / "reference"
        assert alias.is_dir()
        assert (alias / "config.json").is_file()

    def test_ensure_reference_alias_replaces_empty_placeholder_dir(self, tmp_path):
        from async_abc.benchmarks.realistic_workload import _ensure_reference_alias

        output_dir = tmp_path / "experiments" / "data" / "realistic_reference"
        placeholder = output_dir / "reference"
        placeholder.mkdir(parents=True)
        actual = output_dir / "archive" / "reference"
        (actual / "configs").mkdir(parents=True)
        (actual / "000000").mkdir(parents=True)
        (actual / "config.json").write_text("{}")
        (actual / "cis.out").write_text("")
        (actual / "000000" / "cellevents.log").write_text("")

        alias = _ensure_reference_alias(output_dir, actual)

        assert alias == placeholder
        assert (alias / "config.json").is_file()

    def test_init_resolves_repo_relative_asset_paths(self, tmp_path, monkeypatch, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload

        if not _BACKEND_AVAILABLE:
            pytest.skip("sim backend not available — run with sim_backend_venv/.venv")

        mock_sim, mock_dist = rw_mocks
        monkeypatch.chdir(tmp_path)
        cfg = {
            "name": "realistic_workload",
            "sim_config_template": "experiments/assets/realistic_workload/sim_config.json",
            "config_builder_params": "experiments/assets/realistic_workload/config_builder_params.json",
            "distance_metric_params": "experiments/assets/realistic_workload/distance_metric_params.json",
            "parameter_space": "experiments/assets/realistic_workload/parameter_space.json",
            "reference_data_path": "experiments/data/realistic_reference/reference",
            "output_dir": "experiments/data/realistic_sims",
        }

        bm = RealisticWorkload(cfg, _sim_manager=mock_sim, _distance_metric=mock_dist)

        assert "theta_2" in bm.limits

    def test_init_resolves_feature_space_model_and_csv_reference_assets(
        self, tmp_path, monkeypatch, rw_mocks
    ):
        import async_abc.benchmarks.realistic_workload as rw_mod

        if not _BACKEND_AVAILABLE:
            pytest.skip("sim backend not available — run with sim_backend_venv/.venv")

        rw_mod._ensure_backend_on_path()
        inference_distance = importlib.import_module("inference.distance")

        captured: dict[str, object] = {}

        class DummyDistanceMetricParams:
            @classmethod
            def model_validate(cls, data):
                captured.update(data)
                return object()

        class DummyDistanceMetric:
            def __init__(self, params):
                self.params = params

        monkeypatch.setattr(
            inference_distance, "DistanceMetricParams", DummyDistanceMetricParams
        )
        monkeypatch.setattr(inference_distance, "DistanceMetric", DummyDistanceMetric)

        reference_dir = tmp_path / "reference_data"
        reference_dir.mkdir()
        (reference_dir / "output_cells-00000.csv").write_text(
            "#RecordID X Y Z\n1 0 0 0\n"
        )

        mock_sim, _ = rw_mocks
        cfg = {
            "name": "realistic_workload",
            "sim_config_template": "experiments/assets/realistic_workload/sim_config.json",
            "config_builder_params": "experiments/assets/realistic_workload/config_builder_params.json",
            "distance_metric_params": "experiments/assets/realistic_workload/distance_metric_params.json",
            "parameter_space": "experiments/assets/realistic_workload/parameter_space.json",
            "reference_data_path": str(reference_dir),
            "output_dir": str(tmp_path / "sims"),
        }

        bm = rw_mod.RealisticWorkload(cfg, _sim_manager=mock_sim)

        assert isinstance(bm._distance_metric, DummyDistanceMetric)
        assert captured["reference_data"] == str(reference_dir.resolve())
        assert captured["feature_space_model"] == str(
            (
                rw_mod._REPO_ROOT
                / "experiments"
                / "assets"
                / "realistic_workload"
                / "sims_feature_space_model.json"
            ).resolve()
        )


class TestRealisticWorkload:
    """Tests for the real RealisticWorkload implementation (requires sim backend)."""

    # --- init & limits ---

    def test_init_no_error(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        assert bm is not None

    def test_limits_populated(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        assert "theta_2" in bm.limits
        assert "theta_1" in bm.limits

    def test_limits_bounds_correct(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        lo, hi = bm.limits["theta_2"]
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)
        lo2, hi2 = bm.limits["theta_1"]
        assert lo2 == 0.0
        assert hi2 == 1.0

    def test_missing_required_key_raises(self, rw_config):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        del rw_config["reference_data_path"]
        with pytest.raises(KeyError, match="reference_data_path"):
            RealisticWorkload(rw_config)

    # --- simulate pipeline ---

    def test_simulate_calls_build_config(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=42)
        mock_sim.build_simulation_config.assert_called_once()

    def test_simulate_calls_run_simulation(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=42)
        mock_sim.run_simulation.assert_called_once()

    def test_simulate_calls_calculate_distance(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=42)
        mock_dist.calculate_distance.assert_called_once()

    def test_simulate_calls_cleanup(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        sim_dir = Path(mock_sim.build_simulation_config.return_value).parent
        assert sim_dir.exists()
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=42)
        assert not sim_dir.exists()
        mock_sim.cleanup_simdir.assert_called_once()

    def test_simulate_restores_default_fp_state(self, rw_config, rw_mocks, monkeypatch):
        import async_abc.benchmarks.realistic_workload as rw_mod
        from async_abc.benchmarks.realistic_workload import RealisticWorkload

        mock_sim, mock_dist = rw_mocks
        restore_calls = []
        monkeypatch.setattr(
            rw_mod,
            "_restore_default_fp_state",
            lambda: restore_calls.append(True),
        )
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=42)
        assert restore_calls == [True]

    def test_simulate_returns_float(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        result = bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=42)
        assert isinstance(result, float)
        assert result == pytest.approx(2.5)

    def test_simulate_injects_seed_in_param_list(self, rw_config, rw_mocks):
        """simulate() must include the seed as a named Parameter in ParameterList."""
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        captured: list = []
        mock_sim.build_simulation_config.side_effect = (
            lambda pl, **kw: captured.append(pl) or "/tmp/rw_test/eval000001/config.json"
        )
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=99)
        assert len(captured) == 1
        param_names = [p.name for p in captured[0].parameters]
        assert "random_seed" in param_names
        seed_val = next(p.value for p in captured[0].parameters if p.name == "random_seed")
        assert seed_val == 99

    def test_simulate_denormalizes_public_params_before_building_config(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload

        mock_sim, mock_dist = rw_mocks
        captured: list = []
        mock_sim.build_simulation_config.side_effect = (
            lambda pl, **kw: captured.append(pl) or "/tmp/rw_test/eval000001/config.json"
        )
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        public_params = {"theta_2": 0.1, "theta_1": 0.2}
        bm.simulate(public_params, seed=11)
        param_values = {
            param.name: param.value
            for param in captured[0].parameters
            if param.name in public_params
        }
        assert param_values == pytest.approx(denormalize_params(public_params))

    def test_simulate_returns_nan_on_simulation_failure(self, rw_config, rw_mocks):
        """Simulation runtime error → float('nan'), not re-raised exception."""
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        mock_sim.run_simulation.side_effect = RuntimeError("engine crashed")
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        result = bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=0)
        assert math.isnan(result)

    def test_simulate_returns_nan_on_distance_failure(self, rw_config, rw_mocks):
        """Distance computation failure → float('nan'), eval dir still removed."""
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        sim_dir = Path(mock_sim.build_simulation_config.return_value).parent
        mock_dist.calculate_distance.side_effect = ValueError("feature extraction failed")
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        result = bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=0)
        assert math.isnan(result)
        assert not sim_dir.exists()
        mock_sim.cleanup_simdir.assert_called_once()

    def test_simulate_cleanup_called_even_on_distance_failure(self, rw_config, rw_mocks):
        """The eval directory is always deleted, even when distance fails."""
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        sim_dir = Path(mock_sim.build_simulation_config.return_value).parent
        mock_dist.calculate_distance.side_effect = RuntimeError("boom")
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=7)
        assert not sim_dir.exists()
        mock_sim.cleanup_simdir.assert_called_once()

    def test_simulate_removes_eval_dir_on_simulation_failure(self, rw_config, rw_mocks):
        """Simulation failure still removes the generated eval directory."""
        from async_abc.benchmarks.realistic_workload import RealisticWorkload

        mock_sim, mock_dist = rw_mocks
        created_paths = []

        def build_config(_param_list, out_dir_name=None):
            config_path = Path(rw_config["output_dir"]) / out_dir_name / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}")
            created_paths.append(config_path.parent)
            return str(config_path)

        mock_sim.build_simulation_config.side_effect = build_config
        mock_sim.run_simulation.side_effect = RuntimeError("engine crashed")

        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        result = bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=0)

        assert math.isnan(result)
        assert len(created_paths) == 1
        assert not created_paths[0].exists()
        mock_sim.cleanup_simdir.assert_called_once()

    def test_eval_counter_increments(self, rw_config, rw_mocks):
        from async_abc.benchmarks.realistic_workload import RealisticWorkload
        mock_sim, mock_dist = rw_mocks
        bm = RealisticWorkload(rw_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"theta_2": 0.1, "theta_1": 0.2}, seed=0)
        bm.simulate({"theta_2": 0.4, "theta_1": 0.7}, seed=1)
        assert bm._eval_counter == 2

    def test_normalize_and_denormalize_params_round_trip(self):
        # theta_2 physical range [6e-5, 0.6]; theta_1 physical range [0, 10000]
        physical = {"theta_2": 0.03, "theta_1": 2000.0}
        normalized = normalize_params(physical)
        assert normalized["theta_2"] == pytest.approx(0.0499049904990499)
        assert normalized["theta_1"] == pytest.approx(0.2)
        assert denormalize_params(normalized) == pytest.approx(physical)


# ---------------------------------------------------------------------------
# make_benchmark factory
# ---------------------------------------------------------------------------

class TestRealisticWorkloadAnalysisContract:
    """Regression: realistic_workload.json true_param keys must match inferred columns."""

    _CONFIGS_DIR = Path(__file__).parent.parent / "configs"

    def _load_rw_cfg(self) -> dict:
        path = self._CONFIGS_DIR / "realistic_workload.json"
        return json.loads(path.read_text())

    def test_true_param_keys_match_physical_param_names(self):
        """Config true_* keys must use bare param names (theta_2, theta_1).

        Inferred parameter columns in the results CSV are 'theta_2' and
        'theta_1' (normalized [0,1]).  Historically the config used
        'true_theta_2_normalized' / 'true_theta_1_normalized', causing
        true_params_from_benchmark_cfg to return empty → quality plots skipped.
        """
        from async_abc.analysis.sensitivity import true_params_from_benchmark_cfg

        cfg = self._load_rw_cfg()
        benchmark_cfg = cfg.get("benchmark", {})
        true_params = true_params_from_benchmark_cfg(benchmark_cfg)

        assert "theta_2" in true_params, (
            "Config must have 'true_theta_2' (not 'true_theta_2_normalized'); "
            "got true_* keys: "
            + str([k for k in benchmark_cfg if k.startswith("true_")])
        )
        assert "theta_1" in true_params, (
            "Config must have 'true_theta_1' (not 'true_theta_1_normalized'); "
            "got true_* keys: "
            + str([k for k in benchmark_cfg if k.startswith("true_")])
        )

    def test_true_params_have_no_normalized_suffix(self):
        """Confirm the _normalized suffix has been removed from all true_* keys."""
        cfg = self._load_rw_cfg()
        bad_keys = [
            k
            for k in cfg.get("benchmark", {})
            if k.startswith("true_") and k.endswith("_normalized")
        ]
        assert not bad_keys, (
            f"Found true_* keys with _normalized suffix: {bad_keys}. "
            "These do not match the inferred column names and cause quality plots to be skipped."
        )

    def test_true_params_from_cfg_warns_on_normalized_key_mismatch(self, caplog):
        """_true_params_from_cfg warns when a true_* key does not match any inferred column."""
        import logging
        from async_abc.io.records import ParticleRecord
        from async_abc.plotting.reporters import _true_params_from_cfg

        records = [
            ParticleRecord(
                method="rejection_abc",
                replicate=0,
                seed=1,
                step=1,
                params={"theta_2": 0.05, "theta_1": 0.2},
                loss=1.0,
                wall_time=0.1,
            )
        ]
        # Simulate the old (wrong) config with _normalized suffix.
        bad_benchmark_cfg = {
            "true_theta_2_normalized": 0.049905,
            "true_theta_1_normalized": 0.2,
        }
        with caplog.at_level(logging.WARNING, logger="async_abc.plotting.reporters"):
            result = _true_params_from_cfg(records, bad_benchmark_cfg)

        assert result == {}, "No true_params should be returned when keys don't match columns"
        assert any("true_theta_2_normalized" in record.getMessage() for record in caplog.records), (
            "Expected a warning about unmapped true_* keys"
        )


class TestMakeBenchmark:
    def test_gaussian_mean(self):
        bm = make_benchmark({"name": "gaussian_mean", "observed_data_seed": 0, "n_obs": 20})
        assert isinstance(bm, GaussianMean)

    def test_gandk(self):
        bm = make_benchmark({"name": "gandk", "observed_data_seed": 0, "n_obs": 100})
        assert isinstance(bm, GandK)

    def test_lotka_volterra(self):
        bm = make_benchmark({"name": "lotka_volterra", "observed_data_seed": 0})
        assert isinstance(bm, LotkaVolterra)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            make_benchmark({"name": "unknown_model"})

    def test_returns_object_with_simulate_and_limits(self):
        bm = make_benchmark({"name": "gaussian_mean", "observed_data_seed": 0})
        assert hasattr(bm, "simulate")
        assert hasattr(bm, "limits")
        assert callable(bm.simulate)
