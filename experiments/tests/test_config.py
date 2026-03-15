"""Tests for async_abc.io.config and async_abc.io.schema."""
import json

import pytest

from async_abc.io.config import load_config, ValidationError


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
        assert cfg["inference"]["n_workers"] <= 8

    def test_test_mode_clamps_max_simulations(self, config_file):
        cfg = load_config(config_file, test_mode=True)
        assert cfg["inference"]["max_simulations"] <= 500

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
        assert cfg_test["inference"]["n_workers"] <= 8

    def test_large_n_workers_clamped(self, tmp_path, minimal_config):
        minimal_config["inference"]["n_workers"] = 256
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_workers"] == 8

    def test_small_n_workers_unchanged(self, tmp_path, minimal_config):
        minimal_config["inference"]["n_workers"] = 2
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["n_workers"] == 2
