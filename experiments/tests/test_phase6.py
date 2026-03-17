"""Tests for Phase 6: configs + run_all orchestration."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).parent.parent
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RUN_ALL = EXPERIMENTS_DIR / "run_all_paper_experiments.py"
PYTHON = sys.executable

CONFIG_FILES = [
    "gaussian_mean.json",
    "gandk.json",
    "lotka_volterra.json",
    "runtime_heterogeneity.json",
    "scaling.json",
    "sensitivity.json",
    "ablation.json",
]

# Map experiment name → expected output directory name (matches experiment_name in config)
EXPERIMENT_NAMES = [
    "gaussian_mean",
    "gandk",
    "lotka_volterra",
    "runtime_heterogeneity",
    "scaling",
    "sensitivity",
    "ablation",
]


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


# ---------------------------------------------------------------------------
# run_all_paper_experiments.py tests
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_run_all_script_exists(self):
        assert RUN_ALL.exists(), "run_all_paper_experiments.py not found"

    def test_run_all_test_mode_gaussian(self, tmp_path):
        """run_all with --test and --experiments gaussian_mean completes without error."""
        result = subprocess.run(
            [
                PYTHON, str(RUN_ALL),
                "--test",
                "--experiments", "gaussian_mean",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"run_all failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_run_all_creates_output_dir(self, tmp_path):
        """Expected output directory is created for the selected experiment."""
        subprocess.run(
            [
                PYTHON, str(RUN_ALL),
                "--test",
                "--experiments", "gaussian_mean",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        exp_dir = tmp_path / "gaussian_mean"
        assert exp_dir.exists()
        assert (exp_dir / "data").exists()

    def test_run_all_creates_metadata(self, tmp_path):
        """metadata.json should be produced inside data/."""
        subprocess.run(
            [
                PYTHON, str(RUN_ALL),
                "--test",
                "--experiments", "gaussian_mean",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        meta = tmp_path / "gaussian_mean" / "data" / "metadata.json"
        assert meta.exists()

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
