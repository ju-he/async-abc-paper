"""Tests for async_abc.io.config and async_abc.io.schema."""
import json

import pytest

from async_abc.io.config import load_config, ValidationError
from async_abc.io.schema import (
    CLUSTER_TEST_MAX_WORKERS,
    LOCAL_TEST_MAX_WORKERS,
    TEST_MAX_WORKERS_ENV_VAR,
    get_test_mode_overrides,
)
from async_abc.utils.runner import compute_scaling_factor


class TestLoadConfig:
    def test_valid_config(self, config_file, minimal_config):
        cfg = load_config(config_file)
        assert cfg["experiment_name"] == minimal_config["experiment_name"]
        assert cfg["benchmark"]["name"] == "gaussian_mean"

    def test_returns_dict(self, config_file):
        cfg = load_config(config_file)
        assert isinstance(cfg, dict)

    def test_missing_required_key_raises(self, tmp_path):
        bad = {"experiment_name": "x"}  # missing benchmark, methods, inference, execution
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValidationError):
            load_config(p)

    def test_missing_experiment_name_raises(self, tmp_path, minimal_config):
        del minimal_config["experiment_name"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="experiment_name"):
            load_config(p)

    def test_missing_benchmark_raises(self, tmp_path, minimal_config):
        del minimal_config["benchmark"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="benchmark"):
            load_config(p)

    def test_missing_methods_raises(self, tmp_path, minimal_config):
        del minimal_config["methods"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="methods"):
            load_config(p)

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        with pytest.raises(Exception):
            load_config(p)

    def test_test_mode_clamps_n_workers(self, config_file):
        cfg = load_config(config_file, test_mode=True)
        assert cfg["inference"]["n_workers"] <= get_test_mode_overrides()["clamp"]["inference"]["n_workers"]

    def test_test_mode_clamps_max_simulations(self, config_file):
        cfg = load_config(config_file, test_mode=True)
        assert cfg["inference"]["max_simulations"] <= 300

    def test_test_mode_clamps_n_replicates(self, config_file):
        cfg = load_config(config_file, test_mode=True)
        assert cfg["execution"]["n_replicates"] <= 2

    def test_test_mode_sets_single_seed(self, config_file):
        cfg = load_config(config_file, test_mode=True)
        assert cfg["execution"]["base_seed"] == 1

    def test_test_mode_does_not_mutate_original(self, config_file, minimal_config):
        """load_config with test_mode must not mutate the input file."""
        cfg_normal = load_config(config_file)
        cfg_test = load_config(config_file, test_mode=True)
        # Normal mode keeps original values
        assert cfg_normal["inference"]["n_workers"] == minimal_config["inference"]["n_workers"]
        # Test mode has reduced values
        assert cfg_test["inference"]["n_workers"] <= get_test_mode_overrides()["clamp"]["inference"]["n_workers"]

    def test_large_n_workers_clamped_locally(self, tmp_path, minimal_config, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        monkeypatch.delenv(TEST_MAX_WORKERS_ENV_VAR, raising=False)
        minimal_config["inference"]["n_workers"] = 256
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_workers"] == LOCAL_TEST_MAX_WORKERS

    def test_large_n_workers_clamped_on_slurm(self, tmp_path, minimal_config, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        monkeypatch.delenv(TEST_MAX_WORKERS_ENV_VAR, raising=False)
        minimal_config["inference"]["n_workers"] = 256
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_workers"] == CLUSTER_TEST_MAX_WORKERS

    def test_test_mode_worker_cap_env_override(self, tmp_path, minimal_config, monkeypatch):
        monkeypatch.setenv(TEST_MAX_WORKERS_ENV_VAR, "12")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        minimal_config["inference"]["n_workers"] = 256
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_workers"] == 12

    def test_small_n_workers_unchanged(self, tmp_path, minimal_config):
        minimal_config["inference"]["n_workers"] = 2
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_workers"] == 2


class TestNewMethodsInConfig:
    def test_rejection_abc_validates(self, tmp_path, minimal_config):
        minimal_config["methods"] = ["async_propulate_abc", "rejection_abc"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p)
        assert "rejection_abc" in cfg["methods"]

    def test_abc_smc_baseline_validates(self, tmp_path, minimal_config):
        minimal_config["methods"] = ["async_propulate_abc", "abc_smc_baseline"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p)
        assert "abc_smc_baseline" in cfg["methods"]

    def test_all_four_methods_validate(self, tmp_path, minimal_config):
        minimal_config["methods"] = [
            "async_propulate_abc", "pyabc_smc", "rejection_abc", "abc_smc_baseline"
        ]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p)
        assert len(cfg["methods"]) == 4

    def test_n_generations_clamped_in_test_mode(self, tmp_path, minimal_config):
        minimal_config["inference"]["n_generations"] = 10
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_generations"] <= 3

    def test_n_generations_set_when_absent_in_test_mode(self, tmp_path, minimal_config):
        # n_generations not in config — test mode should inject the clamped default
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_generations"] == 3


# ---------------------------------------------------------------------------
# CellularPotts benchmark config validation
# ---------------------------------------------------------------------------

_CPM_BENCHMARK_FULL = {
    "name": "cellular_potts",
    "nastja_config_template": "/some/sim_config.json",
    "config_builder_params": "/some/config_builder_params.json",
    "distance_metric_params": "/some/distance_metric_params.json",
    "parameter_space": "/some/parameter_space.json",
    "reference_data_path": "/some/reference",
    "output_dir": "/some/sims",
}


class TestCPMConfigValidation:
    def test_cpm_config_valid_passes(self, tmp_path, minimal_config):
        minimal_config["benchmark"] = dict(_CPM_BENCHMARK_FULL)
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p)
        assert cfg["benchmark"]["name"] == "cellular_potts"

    def test_cpm_missing_nastja_config_template_raises(self, tmp_path, minimal_config):
        bm = dict(_CPM_BENCHMARK_FULL)
        del bm["nastja_config_template"]
        minimal_config["benchmark"] = bm
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="nastja_config_template"):
            load_config(p)

    def test_cpm_missing_reference_data_path_raises(self, tmp_path, minimal_config):
        bm = dict(_CPM_BENCHMARK_FULL)
        del bm["reference_data_path"]
        minimal_config["benchmark"] = bm
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="reference_data_path"):
            load_config(p)

    def test_cpm_missing_output_dir_raises(self, tmp_path, minimal_config):
        bm = dict(_CPM_BENCHMARK_FULL)
        del bm["output_dir"]
        minimal_config["benchmark"] = bm
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="output_dir"):
            load_config(p)

    def test_non_cpm_benchmark_unaffected(self, tmp_path, minimal_config):
        """gaussian_mean config must not be affected by CPM validation."""
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p)
        assert cfg["benchmark"]["name"] == "gaussian_mean"


class TestSensitivityConfig:
    def test_sensitivity_config_accepts_tol_init_multiplier(self):
        cfg = load_config("configs/sensitivity.json")
        assert "tol_init_multiplier" in cfg["sensitivity_grid"]


class TestScalingFactor:
    def test_sbc_uses_clamped_test_trial_count(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["execution"]["n_replicates"] = 2
        minimal_config["methods"] = ["async_propulate_abc"]
        minimal_config["sbc"] = {"n_trials": 200, "coverage_levels": [0.5, 0.9]}
        path = tmp_path / "sbc.json"
        path.write_text(json.dumps(minimal_config))

        factor, extra, note = compute_scaling_factor(path)

        assert factor == pytest.approx(200 / 3)
        assert extra == 0.0
        assert "200 SBC trials" in note

    def test_straggler_does_not_double_count_slowdown_levels(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["execution"]["n_replicates"] = 2
        minimal_config["straggler"] = {
            "slowdown_factor": [1, 5],
            "base_sleep_s": 0.1,
        }
        path = tmp_path / "straggler.json"
        path.write_text(json.dumps(minimal_config))

        factor, extra, note = compute_scaling_factor(path)

        assert factor == pytest.approx(1.0)
        assert extra > 0.0
        assert "2 slowdown levels" in note

    def test_straggler_scaling_accounts_for_num_methods(self, tmp_path, minimal_config):
        """extra_seconds must scale with the number of configured methods."""
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["n_workers"] = 1
        minimal_config["execution"]["n_replicates"] = 1
        minimal_config["straggler"] = {
            "slowdown_factor": [1],
            "base_sleep_s": 1.0,
        }

        # One method
        minimal_config["methods"] = ["async_propulate_abc"]
        path1 = tmp_path / "straggler_1method.json"
        path1.write_text(json.dumps(minimal_config))
        _, extra_1m, _ = compute_scaling_factor(path1)

        # Two methods — extra should be exactly 2×
        minimal_config["methods"] = ["async_propulate_abc", "abc_smc_baseline"]
        path2 = tmp_path / "straggler_2methods.json"
        path2.write_text(json.dumps(minimal_config))
        _, extra_2m, _ = compute_scaling_factor(path2)

        assert extra_2m == pytest.approx(2.0 * extra_1m)
