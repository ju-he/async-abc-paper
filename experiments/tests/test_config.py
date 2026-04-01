"""Tests for async_abc.io.config and async_abc.io.schema."""
import json

import pytest

from async_abc.io.config import get_run_mode, load_config, ValidationError
from async_abc.io.schema import (
    CLUSTER_TEST_MAX_WORKERS,
    LOCAL_TEST_MAX_WORKERS,
    TEST_MAX_WORKERS_ENV_VAR,
    get_test_mode_overrides,
)
from async_abc.utils.runner import compute_corrected_estimate, compute_scaling_factor


class TestLoadConfig:
    def test_valid_config(self, config_file, minimal_config):
        cfg = load_config(config_file)
        assert cfg["experiment_name"] == minimal_config["experiment_name"]
        assert cfg["benchmark"]["name"] == "gaussian_mean"
        assert cfg["inference"]["progress_log_interval_s"] == 10.0

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

    def test_small_mode_loads_sibling_config(self, tmp_path, minimal_config):
        base_path = tmp_path / "cfg.json"
        base_path.write_text(json.dumps(minimal_config))
        small_dir = tmp_path / "small"
        small_dir.mkdir()
        small_cfg = json.loads(json.dumps(minimal_config))
        small_cfg["inference"]["max_simulations"] = 123
        (small_dir / "cfg.json").write_text(json.dumps(small_cfg))

        cfg = load_config(base_path, small_mode=True)

        assert cfg["inference"]["max_simulations"] == 123
        assert cfg["execution"]["config_tier"] == "small"
        assert get_run_mode(cfg) == "small"
        assert cfg["inference"]["test_mode"] is False

    def test_small_mode_then_test_mode_stacks(self, tmp_path, minimal_config):
        base_path = tmp_path / "cfg.json"
        base_path.write_text(json.dumps(minimal_config))
        small_dir = tmp_path / "small"
        small_dir.mkdir()
        small_cfg = json.loads(json.dumps(minimal_config))
        small_cfg["inference"]["max_simulations"] = 250
        small_cfg["execution"]["n_replicates"] = 2
        (small_dir / "cfg.json").write_text(json.dumps(small_cfg))

        cfg = load_config(base_path, small_mode=True, test_mode=True)

        assert cfg["inference"]["max_simulations"] <= 100
        assert cfg["execution"]["config_tier"] == "small"
        assert get_run_mode(cfg) == "small_test"
        assert cfg["inference"]["test_mode"] is True

    def test_small_mode_missing_sibling_raises(self, tmp_path, minimal_config):
        base_path = tmp_path / "cfg.json"
        base_path.write_text(json.dumps(minimal_config))

        with pytest.raises(FileNotFoundError, match="Small config not found"):
            load_config(base_path, small_mode=True)

    def test_preserves_custom_progress_interval(self, tmp_path, minimal_config):
        minimal_config["inference"]["progress_log_interval_s"] = 2.5
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))

        cfg = load_config(p)

        assert cfg["inference"]["progress_log_interval_s"] == 2.5


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
        assert cfg["inference"]["n_generations"] == 2


class TestNGenerationsSafetyNet:
    """When wall-time is the primary stop, n_generations must not be the binding constraint."""

    def test_config_with_wall_time_warns_low_n_generations(self, tmp_path, minimal_config):
        """Low n_generations with max_wall_time_s should emit a warning."""
        import warnings

        minimal_config["inference"]["max_wall_time_s"] = 300
        minimal_config["inference"]["n_generations"] = 5
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(p)
            n_gen_warnings = [x for x in w if "n_generations" in str(x.message)]
            assert len(n_gen_warnings) >= 1

    def test_config_wall_time_sets_default_high_n_generations(self, tmp_path, minimal_config):
        """When max_wall_time_s is set and n_generations is absent, default to 1000."""
        minimal_config["inference"]["max_wall_time_s"] = 300
        # n_generations not set — should default high
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p)
        assert cfg["inference"].get("n_generations", 5) >= 1000

    def test_no_warning_without_wall_time(self, tmp_path, minimal_config):
        """Without max_wall_time_s, low n_generations should not warn."""
        import warnings

        minimal_config["inference"]["n_generations"] = 5
        # no max_wall_time_s
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(p)
            n_gen_warnings = [x for x in w if "n_generations" in str(x.message)]
            assert len(n_gen_warnings) == 0


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

    def test_cpm_test_mode_uses_smaller_budget(self, tmp_path, minimal_config):
        minimal_config["benchmark"] = dict(_CPM_BENCHMARK_FULL)
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["k"] = 100
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))

        cfg = load_config(p, test_mode=True)

        assert cfg["inference"]["test_mode"] is True
        assert cfg["inference"]["max_simulations"] <= 12
        assert cfg["inference"]["k"] <= 5


class TestWallTimeClamping:
    """Test-mode clamping for max_wall_time_s."""

    def test_test_mode_clamps_max_wall_time_s(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_wall_time_s"] = 600
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["max_wall_time_s"] <= 30

    def test_test_mode_injects_wall_time_when_absent(self, tmp_path, minimal_config):
        """If max_wall_time_s is absent, test mode injects the clamp default (30s)."""
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p, test_mode=True)
        assert cfg["inference"]["max_wall_time_s"] == 30

    @pytest.mark.parametrize("config_name", [
        "gaussian_mean.json",
        "gandk.json",
        "lotka_volterra.json",
        "cellular_potts.json",
    ])
    def test_all_benchmark_configs_have_wall_time(self, config_name):
        cfg = load_config(f"configs/{config_name}")
        assert "max_wall_time_s" in cfg["inference"], (
            f"{config_name} missing max_wall_time_s"
        )
        assert cfg["inference"]["max_wall_time_s"] > 0


class TestValueValidation:
    """Validate scheduler_type and benchmark.name against allowed values."""

    def test_invalid_scheduler_type_raises(self, tmp_path, minimal_config):
        minimal_config["inference"]["scheduler_type"] = "nonexistent_scheduler"
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="scheduler_type"):
            load_config(p)

    def test_valid_scheduler_types_accepted(self, tmp_path, minimal_config):
        for stype in ("quantile", "geometric_decay", "acceptance_rate"):
            minimal_config["inference"]["scheduler_type"] = stype
            p = tmp_path / "cfg.json"
            p.write_text(json.dumps(minimal_config))
            cfg = load_config(p)
            assert cfg["inference"]["scheduler_type"] == stype

    def test_invalid_benchmark_name_raises(self, tmp_path, minimal_config):
        minimal_config["benchmark"]["name"] = "nonexistent_benchmark"
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        with pytest.raises(ValidationError, match="benchmark.*name"):
            load_config(p)

    def test_valid_benchmark_names_accepted(self, tmp_path, minimal_config):
        # Only test gaussian_mean here (others need extra benchmark keys)
        minimal_config["benchmark"]["name"] = "gaussian_mean"
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(minimal_config))
        cfg = load_config(p)
        assert cfg["benchmark"]["name"] == "gaussian_mean"


class TestSensitivityConfig:
    def test_sensitivity_config_accepts_tol_init_multiplier(self):
        cfg = load_config("configs/sensitivity.json")
        assert "tol_init_multiplier" in cfg["sensitivity_grid"]

    def test_tol_init_multiplier_has_four_levels(self):
        """Paper-concept specifies [0.5, 1.0, 2.0, 5.0] — all four must be present."""
        cfg = load_config("configs/sensitivity.json")
        levels = cfg["sensitivity_grid"]["tol_init_multiplier"]
        assert 2.0 in levels, f"Missing 2.0 in tol_init_multiplier levels: {levels}"
        assert len(levels) == 4, f"Expected 4 levels, got {len(levels)}: {levels}"

    def test_n_workers_not_in_sensitivity_inference(self):
        """n_workers is a system parameter and must not appear in the inference block."""
        cfg = load_config("configs/sensitivity.json")
        assert "n_workers" not in cfg["inference"], (
            "n_workers should not be in sensitivity inference config"
        )

    def test_sensitivity_gandk_config_exists(self):
        """A sensitivity_gandk.json config must exist for the g-and-k benchmark."""
        # load_config resolves relative to experiments/, so this will raise
        # FileNotFoundError if the file is absent.
        cfg = load_config("configs/sensitivity_gandk.json")
        assert cfg is not None

    def test_sensitivity_gandk_benchmark_is_gandk(self):
        cfg = load_config("configs/sensitivity_gandk.json")
        assert cfg["benchmark"]["name"] == "gandk"

    def test_sensitivity_gandk_has_adequate_budget(self):
        cfg = load_config("configs/sensitivity_gandk.json")
        assert cfg["inference"]["max_simulations"] >= 20000, (
            f"g-and-k sensitivity needs >= 20000 simulations, "
            f"got {cfg['inference']['max_simulations']}"
        )


class TestScalingFactor:
    def test_propulate_total_budget_mode_scales_with_worker_count(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["n_workers"] = 4
        minimal_config["inference"]["propulate_budget_mode"] = "total_simulations"
        path = tmp_path / "propulate_total_budget.json"
        path.write_text(json.dumps(minimal_config))

        factor, extra, note = compute_scaling_factor(path)

        assert factor == pytest.approx(9.0)
        assert extra == 0.0
        assert "300 sims × 3 reps" in note

    def test_sbc_uses_clamped_test_trial_count(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["n_workers"] = 4
        minimal_config["execution"]["n_replicates"] = 2
        minimal_config["methods"] = ["async_propulate_abc"]
        minimal_config["sbc"] = {"n_trials": 200, "coverage_levels": [0.5, 0.9]}
        path = tmp_path / "sbc.json"
        path.write_text(json.dumps(minimal_config))

        factor, extra, note = compute_scaling_factor(path)

        assert factor == pytest.approx(2400.0)
        assert extra == 0.0
        assert "200 SBC trials" in note

    def test_straggler_does_not_double_count_slowdown_levels(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["n_workers"] = 4
        minimal_config["execution"]["n_replicates"] = 2
        minimal_config["straggler"] = {
            "slowdown_factor": [1, 5],
            "base_sleep_s": 0.1,
        }
        path = tmp_path / "straggler.json"
        path.write_text(json.dumps(minimal_config))

        factor, extra, note = compute_scaling_factor(path)

        assert factor == pytest.approx(24.0)
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

    def test_corrected_estimate_uses_method_specific_compute_scale(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["n_workers"] = 4
        minimal_config["execution"]["n_replicates"] = 2
        minimal_config["methods"] = ["async_propulate_abc", "abc_smc_baseline"]
        path = tmp_path / "mixed_methods.json"
        path.write_text(json.dumps(minimal_config))

        raw_results = tmp_path / "raw_results.csv"
        raw_results.write_text(
            "\n".join(
                [
                    "method,wall_time",
                    "async_propulate_abc,4.0",
                    "abc_smc_baseline,6.0",
                ]
            )
            + "\n"
        )

        estimated = compute_corrected_estimate(15.0, raw_results, path)

        assert estimated == pytest.approx(142.0)

    def test_corrected_estimate_counts_multiple_active_replicates(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["n_workers"] = 4
        minimal_config["execution"]["n_replicates"] = 4
        minimal_config["methods"] = ["abc_smc_baseline"]
        path = tmp_path / "mixed_replicates.json"
        path.write_text(json.dumps(minimal_config))

        small_dir = tmp_path / "small"
        small_dir.mkdir()
        small_cfg = json.loads(json.dumps(minimal_config))
        small_cfg["inference"]["max_simulations"] = 150
        small_cfg["execution"]["n_replicates"] = 2
        (small_dir / path.name).write_text(json.dumps(small_cfg))

        raw_results = tmp_path / "raw_results.csv"
        raw_results.write_text(
            "\n".join(
                [
                    "method,replicate,wall_time",
                    "abc_smc_baseline,0,3.0",
                    "abc_smc_baseline,1,5.0",
                ]
            )
            + "\n"
        )

        estimated = compute_corrected_estimate(
            12.0,
            raw_results,
            path,
            small_mode=True,
            test_mode=False,
        )

        assert estimated == pytest.approx(40.0)

    def test_corrected_estimate_single_replicate_column_matches_previous_behavior(self, tmp_path, minimal_config):
        minimal_config["inference"]["max_simulations"] = 300
        minimal_config["inference"]["n_workers"] = 4
        minimal_config["execution"]["n_replicates"] = 2
        minimal_config["methods"] = ["async_propulate_abc", "abc_smc_baseline"]
        path = tmp_path / "single_replicate_column.json"
        path.write_text(json.dumps(minimal_config))

        raw_results = tmp_path / "raw_results.csv"
        raw_results.write_text(
            "\n".join(
                [
                    "method,replicate,wall_time",
                    "async_propulate_abc,0,4.0",
                    "abc_smc_baseline,0,6.0",
                ]
            )
            + "\n"
        )

        estimated = compute_corrected_estimate(15.0, raw_results, path)

        assert estimated == pytest.approx(142.0)
