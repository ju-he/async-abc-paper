"""Tests for Phase 6: configs + run_all orchestration."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers

EXPERIMENTS_DIR = Path(__file__).parent.parent
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RUN_ALL = EXPERIMENTS_DIR / "run_all_paper_experiments.py"
PYTHON = sys.executable

CONFIG_FILES = [
    "gaussian_mean.json",
    "gandk.json",
    "lotka_volterra.json",
    "cellular_potts.json",
    "sbc.json",
    "straggler.json",
    "runtime_heterogeneity.json",
    "scaling.json",
    "sensitivity.json",
    "sensitivity_gandk.json",
    "ablation.json",
]

SMALL_CONFIG_FILES = [f"small/{name}" for name in CONFIG_FILES]

# Map experiment name → expected output directory name (matches experiment_name in config)
EXPERIMENT_NAMES = [
    "gaussian_mean",
    "gandk",
    "lotka_volterra",
    "cellular_potts",
    "sbc",
    "straggler",
    "runtime_heterogeneity",
    "scaling",
    "sensitivity",
    "sensitivity_gandk",
    "ablation",
]


@pytest.fixture(scope="module")
def run_all_gaussian(tmp_path_factory):
    root = tmp_path_factory.mktemp("run_all_gaussian")
    cfg = test_helpers.make_fast_runner_config(
        "gaussian_mean.json",
        methods=["rejection_abc"],
        inference_overrides={"max_simulations": 60, "k": 10},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={},
    )
    config_path = test_helpers.write_config(root, "gaussian_run_all.json", cfg)
    run_all = test_helpers.import_runner_module("../run_all_paper_experiments.py")
    original_configs_dir = run_all.CONFIGS_DIR
    original_registry = run_all.EXPERIMENT_REGISTRY
    try:
        run_all.CONFIGS_DIR = root
        run_all.EXPERIMENT_REGISTRY = {
            "gaussian_mean": ("gaussian_mean_runner.py", config_path.name),
        }
        run_all.main(
            [
                "--test",
                "--experiments",
                "gaussian_mean",
                "--output-dir",
                str(root),
            ]
        )
    finally:
        run_all.CONFIGS_DIR = original_configs_dir
        run_all.EXPERIMENT_REGISTRY = original_registry
    return {"output_dir": root}


# ---------------------------------------------------------------------------
# test_all_configs_valid
# ---------------------------------------------------------------------------

class TestAllConfigsValid:
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_config_file_exists(self, config_file):
        assert (CONFIGS_DIR / config_file).exists(), f"Missing config: {config_file}"

    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_config_passes_schema_validation(self, config_file):
        """Each config must load without ValidationError."""
        sys.path.insert(0, str(EXPERIMENTS_DIR))
        from async_abc.io.config import load_config
        cfg = load_config(CONFIGS_DIR / config_file)
        assert isinstance(cfg, dict)
        assert "experiment_name" in cfg

    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_config_is_valid_json(self, config_file):
        text = (CONFIGS_DIR / config_file).read_text()
        obj = json.loads(text)
        assert isinstance(obj, dict)

    @pytest.mark.parametrize("config_file", SMALL_CONFIG_FILES)
    def test_small_config_file_exists(self, config_file):
        assert (CONFIGS_DIR / config_file).exists(), f"Missing small config: {config_file}"

    @pytest.mark.parametrize("config_file", SMALL_CONFIG_FILES)
    def test_small_config_passes_schema_validation(self, config_file):
        from async_abc.io.config import load_config

        cfg = load_config(CONFIGS_DIR / config_file)
        assert isinstance(cfg, dict)
        assert cfg["execution"]["config_tier"] == "small"


# ---------------------------------------------------------------------------
# run_all_paper_experiments.py tests
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_run_all_script_exists(self):
        assert RUN_ALL.exists(), "run_all_paper_experiments.py not found"

    def test_run_all_registry_includes_all_paper_experiments(self):
        run_all = test_helpers.import_runner_module("../run_all_paper_experiments.py")
        assert set(EXPERIMENT_NAMES).issubset(run_all.EXPERIMENT_REGISTRY)

    def test_run_all_test_mode_gaussian(self, run_all_gaussian):
        """run_all with --test and --experiments gaussian_mean completes without error."""
        exp_dir = run_all_gaussian["output_dir"] / "gaussian_mean"
        assert exp_dir.exists()

    def test_run_all_creates_output_dir(self, run_all_gaussian):
        """Expected output directory is created for the selected experiment."""
        exp_dir = run_all_gaussian["output_dir"] / "gaussian_mean"
        assert exp_dir.exists()
        assert (exp_dir / "data").exists()

    def test_run_all_creates_metadata(self, run_all_gaussian):
        """metadata.json should be produced inside data/."""
        meta = run_all_gaussian["output_dir"] / "gaussian_mean" / "data" / "metadata.json"
        assert meta.exists()

    def test_run_all_metadata_includes_paper_role_and_stop_policy(self, run_all_gaussian):
        meta = json.loads(
            (run_all_gaussian["output_dir"] / "gaussian_mean" / "data" / "metadata.json").read_text()
        )
        assert meta["experiment_role"] == "validity"
        assert meta["stop_policy"] == "fixed_walltime"
        assert meta["method_comparison_roles"]["rejection_abc"] == "small_model_reference"

    def test_run_all_help_flag(self):
        """--help should exit 0 and print usage."""
        result = subprocess.run(
            [PYTHON, str(RUN_ALL), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "output-dir" in result.stdout.lower() or "output-dir" in result.stderr.lower()

    def test_run_all_unknown_experiment_exits_nonzero(self, tmp_path):
        """An unknown experiment name should cause a non-zero exit."""
        result = subprocess.run(
            [
                PYTHON, str(RUN_ALL),
                "--test",
                "--experiments", "nonexistent_experiment",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_run_all_accepts_all_token(self, tmp_path):
        run_all = test_helpers.import_runner_module("../run_all_paper_experiments.py")
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 40, "k": 10},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={},
        )
        config_path = test_helpers.write_config(tmp_path, "gaussian_all.json", cfg)
        original_configs_dir = run_all.CONFIGS_DIR
        original_registry = run_all.EXPERIMENT_REGISTRY
        try:
            run_all.CONFIGS_DIR = tmp_path
            run_all.EXPERIMENT_REGISTRY = {
                "gaussian_mean": ("gaussian_mean_runner.py", config_path.name),
            }
            run_all.main(
                [
                    "--test",
                    "--experiments",
                    "all",
                    "--output-dir",
                    str(tmp_path),
                ]
            )
        finally:
            run_all.CONFIGS_DIR = original_configs_dir
            run_all.EXPERIMENT_REGISTRY = original_registry

        assert (tmp_path / "gaussian_mean" / "data" / "metadata.json").exists()

    def test_run_all_forwards_small_and_test_flags(self, tmp_path, monkeypatch):
        run_all = test_helpers.import_runner_module("../run_all_paper_experiments.py")
        captured = {}

        monkeypatch.setattr(
            run_all,
            "_run_experiment",
            lambda name, runner, config, output_dir, test_mode, small_mode, extend=False: (
                captured.update(
                    {
                        "name": name,
                        "runner": runner,
                        "config": config,
                        "test_mode": test_mode,
                        "small_mode": small_mode,
                        "extend": extend,
                    }
                ) or (0, 0.1)
            ),
        )
        monkeypatch.setattr(run_all, "_read_runner_estimate", lambda _: 1.0)

        run_all.main(
            [
                "--small",
                "--test",
                "--experiments",
                "gaussian_mean",
                "--output-dir",
                str(tmp_path),
            ]
        )

        timing_rows = list(csv.DictReader(open(tmp_path / "timing_summary_small_test.csv")))
        assert captured["small_mode"] is True
        assert captured["test_mode"] is True
        assert timing_rows[-1]["run_mode"] == "small_test"


class TestGaussianMeanConfigMethods:
    def test_gaussian_mean_has_rejection_abc(self):
        from async_abc.io.config import load_config
        cfg = load_config(CONFIGS_DIR / "gaussian_mean.json")
        assert "rejection_abc" in cfg["methods"]

    def test_gaussian_mean_has_abc_smc_baseline(self):
        from async_abc.io.config import load_config
        cfg = load_config(CONFIGS_DIR / "gaussian_mean.json")
        assert "abc_smc_baseline" in cfg["methods"]

    def test_gaussian_mean_has_n_generations(self):
        from async_abc.io.config import load_config
        cfg = load_config(CONFIGS_DIR / "gaussian_mean.json")
        assert "n_generations" in cfg["inference"]

    def test_gandk_has_rejection_abc(self):
        from async_abc.io.config import load_config
        cfg = load_config(CONFIGS_DIR / "gandk.json")
        assert "rejection_abc" in cfg["methods"]

    def test_gandk_has_abc_smc_baseline(self):
        from async_abc.io.config import load_config
        cfg = load_config(CONFIGS_DIR / "gandk.json")
        assert "abc_smc_baseline" in cfg["methods"]


class TestPhase3ConfigPlots:
    @pytest.mark.parametrize("config_file", [
        "gaussian_mean.json",
        "gandk.json",
        "lotka_volterra.json",
    ])
    def test_benchmark_configs_enable_phase3_plots(self, config_file):
        cfg = json.loads((CONFIGS_DIR / config_file).read_text())
        plots = cfg["plots"]
        assert plots.get("corner") is True
        assert plots.get("tolerance_trajectory") is True
        assert plots.get("quality_vs_time") is True

    def test_runtime_heterogeneity_enables_gantt(self):
        cfg = json.loads((CONFIGS_DIR / "runtime_heterogeneity.json").read_text())
        assert cfg["plots"].get("gantt") is True


class TestPhase5ConfigPlots:
    def test_straggler_enables_plots(self):
        cfg = json.loads((CONFIGS_DIR / "straggler.json").read_text())
        assert cfg["plots"].get("throughput_vs_slowdown") is True
        assert cfg["plots"].get("gantt") is True
