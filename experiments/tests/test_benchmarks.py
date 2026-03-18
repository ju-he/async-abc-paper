"""Tests for async_abc.benchmarks.*"""
import builtins
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
from async_abc.benchmarks.cellular_potts import _ensure_nastjapy_on_path

# ---------------------------------------------------------------------------
# CPM helpers & fixtures
# ---------------------------------------------------------------------------

_CPM_TEMPLATE_DIR = Path(__file__).parents[1] / "assets" / "cellular_potts"


def _nastjapy_available() -> bool:
    try:
        _ensure_nastjapy_on_path()
        return True
    except ImportError:
        return False


_NASTJAPY_AVAILABLE = _nastjapy_available()


@pytest.fixture
def cpm_config(tmp_path):
    """Minimal CellularPotts config using real template files (nastjapy must be available)."""
    if not _NASTJAPY_AVAILABLE:
        pytest.skip("nastjapy not available — run with nastjapy_copy/.venv")
    return {
        "name": "cellular_potts",
        "nastja_config_template": str(_CPM_TEMPLATE_DIR / "sim_config.json"),
        "config_builder_params": str(_CPM_TEMPLATE_DIR / "config_builder_params.json"),
        "distance_metric_params": str(_CPM_TEMPLATE_DIR / "distance_metric_params.json"),
        "parameter_space": str(_CPM_TEMPLATE_DIR / "parameter_space_division_motility.json"),
        "reference_data_path": str(tmp_path / "reference"),
        "output_dir": str(tmp_path / "sims"),
    }


@pytest.fixture
def cpm_mocks():
    """Pre-built mock SimulationManager and DistanceMetric for injection."""
    mock_sim = MagicMock()
    mock_dist = MagicMock()
    # Default: build_simulation_config returns a plausible config path
    mock_sim.build_simulation_config.return_value = "/tmp/cpm_test/eval000001/config.json"
    # Default: calculate_distance returns a positive float
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


# ---------------------------------------------------------------------------
# CellularPotts
# ---------------------------------------------------------------------------

class TestCellularPottsImport:
    """Module-level import must always succeed regardless of nastjapy availability."""

    def test_import_does_not_raise(self):
        from async_abc.benchmarks import cellular_potts  # noqa: F401

    def test_class_importable(self):
        from async_abc.benchmarks.cellular_potts import CellularPotts  # noqa: F401

    def test_missing_nastjapy_raises_import_error_on_init(self, tmp_path, monkeypatch):
        """When nastja is unavailable from both env and fallback, __init__ raises ImportError."""
        import async_abc.benchmarks.cellular_potts as cellular_potts

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nastja" or name.startswith("nastja."):
                raise ModuleNotFoundError("No module named 'nastja'")
            return real_import(name, globals, locals, fromlist, level)

        real_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(ImportError):
            cellular_potts.CellularPotts({"name": "cellular_potts"})

    def test_normalize_generated_config_paths_rewrites_repo_relative_include(self, tmp_path, monkeypatch):
        from async_abc.benchmarks.cellular_potts import _normalize_generated_config_paths

        include_target = tmp_path / "shared" / "filling.json"
        include_target.parent.mkdir(parents=True)
        include_target.write_text("{}")
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / "runs" / "reference"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps({"Include": [str(include_target.relative_to(tmp_path))]}))

        normalized = _normalize_generated_config_paths(config_path)
        data = json.loads(normalized.read_text())

        assert data["Include"] == [str(include_target.resolve())]

    def test_resolve_reference_data_path_finds_nested_generated_dir(self, tmp_path, monkeypatch):
        import async_abc.benchmarks.cellular_potts as cellular_potts

        configured = tmp_path / "experiments" / "data" / "cpm_reference" / "reference"
        configured.mkdir(parents=True)
        nested = (
            tmp_path
            / "experiments"
            / "data"
            / "cpm_reference"
            / "experiments"
            / "data"
            / "cpm_reference"
            / "reference"
        )
        (nested / "configs").mkdir(parents=True)
        (nested / "000000").mkdir(parents=True)
        (nested / "config.json").write_text("{}")
        (nested / "cis.out").write_text("")
        (nested / "000000" / "cellevents.log").write_text("")

        monkeypatch.setattr(cellular_potts, "_REPO_ROOT", tmp_path)

        resolved = cellular_potts._resolve_reference_data_path(
            "experiments/data/cpm_reference/reference"
        )

        assert resolved == nested.resolve()

    def test_resolve_reference_data_path_finds_arbitrarily_nested_dir(self, tmp_path, monkeypatch):
        import async_abc.benchmarks.cellular_potts as cellular_potts

        configured = tmp_path / "experiments" / "data" / "cpm_reference" / "reference"
        configured.mkdir(parents=True)
        nested = (
            tmp_path
            / "experiments"
            / "data"
            / "cpm_reference"
            / "archive"
            / "2026-03-18"
            / "reference"
        )
        (nested / "configs").mkdir(parents=True)
        (nested / "000000").mkdir(parents=True)
        (nested / "config.json").write_text("{}")
        (nested / "cis.out").write_text("")
        (nested / "000000" / "cellevents.log").write_text("")

        monkeypatch.setattr(cellular_potts, "_REPO_ROOT", tmp_path)

        resolved = cellular_potts._resolve_reference_data_path(
            "experiments/data/cpm_reference/reference"
        )

        assert resolved == nested.resolve()

    def test_resolve_reference_data_path_accepts_csv_reference_dir(self, tmp_path, monkeypatch):
        import async_abc.benchmarks.cellular_potts as cellular_potts

        configured = tmp_path / "experiments" / "assets" / "cellular_potts" / "reference_data"
        configured.mkdir(parents=True)
        (configured / "output_cells-00000.csv").write_text("#CellID CenterX CenterY CenterZ\n1 0 0 0\n")

        monkeypatch.setattr(cellular_potts, "_REPO_ROOT", tmp_path)

        resolved = cellular_potts._resolve_reference_data_path(
            "experiments/assets/cellular_potts/reference_data"
        )

        assert resolved == configured.resolve()

    def test_ensure_reference_alias_reuses_canonical_reference_path(self, tmp_path):
        from async_abc.benchmarks.cellular_potts import _ensure_reference_alias

        output_dir = tmp_path / "experiments" / "data" / "cpm_reference"
        actual = output_dir / "experiments" / "data" / "cpm_reference" / "reference"
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
        from async_abc.benchmarks.cellular_potts import _ensure_reference_alias

        output_dir = tmp_path / "experiments" / "data" / "cpm_reference"
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

    def test_init_resolves_repo_relative_asset_paths(self, tmp_path, monkeypatch, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts

        if not _NASTJAPY_AVAILABLE:
            pytest.skip("nastjapy not available — run with nastjapy_copy/.venv")

        mock_sim, mock_dist = cpm_mocks
        monkeypatch.chdir(tmp_path)
        cfg = {
            "name": "cellular_potts",
            "nastja_config_template": "experiments/assets/cellular_potts/sim_config.json",
            "config_builder_params": "experiments/assets/cellular_potts/config_builder_params.json",
            "distance_metric_params": "experiments/assets/cellular_potts/distance_metric_params.json",
            "parameter_space": "experiments/assets/cellular_potts/parameter_space_division_motility.json",
            "reference_data_path": "experiments/data/cpm_reference/reference",
            "output_dir": "experiments/data/cpm_sims",
        }

        bm = CellularPotts(cfg, _sim_manager=mock_sim, _distance_metric=mock_dist)

        assert "division_rate" in bm.limits

    def test_init_resolves_feature_space_model_and_csv_reference_assets(
        self, tmp_path, monkeypatch, cpm_mocks
    ):
        import async_abc.benchmarks.cellular_potts as cellular_potts

        if not _NASTJAPY_AVAILABLE:
            pytest.skip("nastjapy not available — run with nastjapy_copy/.venv")

        cellular_potts._ensure_nastjapy_on_path()
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
            "#CellID CenterX CenterY CenterZ\n1 0 0 0\n"
        )

        mock_sim, _ = cpm_mocks
        cfg = {
            "name": "cellular_potts",
            "nastja_config_template": "experiments/assets/cellular_potts/sim_config.json",
            "config_builder_params": "experiments/assets/cellular_potts/config_builder_params.json",
            "distance_metric_params": "experiments/assets/cellular_potts/distance_metric_params.json",
            "parameter_space": "experiments/assets/cellular_potts/parameter_space_division_motility.json",
            "reference_data_path": str(reference_dir),
            "output_dir": str(tmp_path / "sims"),
        }

        bm = cellular_potts.CellularPotts(cfg, _sim_manager=mock_sim)

        assert isinstance(bm._distance_metric, DummyDistanceMetric)
        assert captured["reference_data"] == str(reference_dir.resolve())
        assert captured["feature_space_model"] == str(
            (
                cellular_potts._REPO_ROOT
                / "experiments"
                / "assets"
                / "cellular_potts"
                / "sims_feature_space_model.json"
            ).resolve()
        )


class TestCellularPotts:
    """Tests for the real CellularPotts implementation (requires nastjapy venv)."""

    # --- init & limits ---

    def test_init_no_error(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        assert bm is not None

    def test_limits_populated(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        assert "division_rate" in bm.limits
        assert "motility" in bm.limits

    def test_limits_bounds_correct(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        lo, hi = bm.limits["division_rate"]
        assert lo == pytest.approx(0.00006)
        assert hi == pytest.approx(0.6)
        lo2, hi2 = bm.limits["motility"]
        assert lo2 == 0.0
        assert hi2 == 10000.0

    def test_missing_required_key_raises(self, cpm_config):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        del cpm_config["reference_data_path"]
        with pytest.raises(KeyError, match="reference_data_path"):
            CellularPotts(cpm_config)

    # --- simulate pipeline ---

    def test_simulate_calls_build_config(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=42)
        mock_sim.build_simulation_config.assert_called_once()

    def test_simulate_calls_run_simulation(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=42)
        mock_sim.run_simulation.assert_called_once()

    def test_simulate_calls_calculate_distance(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=42)
        mock_dist.calculate_distance.assert_called_once()

    def test_simulate_calls_cleanup(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=42)
        mock_sim.cleanup_simdir.assert_called_once()

    def test_simulate_returns_float(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        result = bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=42)
        assert isinstance(result, float)
        assert result == pytest.approx(2.5)

    def test_simulate_injects_seed_in_param_list(self, cpm_config, cpm_mocks):
        """simulate() must include the seed as a named Parameter in ParameterList."""
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        captured: list = []
        mock_sim.build_simulation_config.side_effect = (
            lambda pl, **kw: captured.append(pl) or "/tmp/cpm_test/eval000001/config.json"
        )
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=99)
        assert len(captured) == 1
        param_names = [p.name for p in captured[0].parameters]
        assert "random_seed" in param_names
        seed_val = next(p.value for p in captured[0].parameters if p.name == "random_seed")
        assert seed_val == 99

    def test_simulate_returns_nan_on_simulation_failure(self, cpm_config, cpm_mocks):
        """Simulation runtime error → float('nan'), not re-raised exception."""
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        mock_sim.run_simulation.side_effect = RuntimeError("NAStJA crashed")
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        result = bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=0)
        assert math.isnan(result)

    def test_simulate_returns_nan_on_distance_failure(self, cpm_config, cpm_mocks):
        """Distance computation failure → float('nan'), cleanup still called."""
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        mock_dist.calculate_distance.side_effect = ValueError("feature extraction failed")
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        result = bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=0)
        assert math.isnan(result)
        mock_sim.cleanup_simdir.assert_called_once()

    def test_simulate_cleanup_called_even_on_distance_failure(self, cpm_config, cpm_mocks):
        """cleanup_simdir is always called (finally block), even when distance fails."""
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        mock_dist.calculate_distance.side_effect = RuntimeError("boom")
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=7)
        mock_sim.cleanup_simdir.assert_called_once()

    def test_eval_counter_increments(self, cpm_config, cpm_mocks):
        from async_abc.benchmarks.cellular_potts import CellularPotts
        mock_sim, mock_dist = cpm_mocks
        bm = CellularPotts(cpm_config, _sim_manager=mock_sim, _distance_metric=mock_dist)
        bm.simulate({"division_rate": 0.01, "motility": 500.0}, seed=0)
        bm.simulate({"division_rate": 0.1, "motility": 1000.0}, seed=1)
        assert bm._eval_counter == 2


# ---------------------------------------------------------------------------
# make_benchmark factory
# ---------------------------------------------------------------------------

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
