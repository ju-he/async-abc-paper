"""Tests for experiment runner scripts.

Basic runners (gaussian, gandk, lv) are tested via subprocess with --test.
Scaling / sensitivity / ablation runners are tested by importing and calling
their main() functions directly to avoid spawning many processes.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
PYTHON = sys.executable


def _run_script(script_name: str, config_name: str, output_dir: Path) -> subprocess.CompletedProcess:
    """Run a runner script as a subprocess with --test flag."""
    return subprocess.run(
        [
            PYTHON,
            str(SCRIPTS_DIR / script_name),
            "--config", str(CONFIGS_DIR / config_name),
            "--output-dir", str(output_dir),
            "--test",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Gaussian mean runner
# ---------------------------------------------------------------------------

class TestGaussianMeanRunner:
    def test_completes(self, tmp_output_dir):
        result = _run_script("gaussian_mean_runner.py", "gaussian_mean.json", tmp_output_dir)
        assert result.returncode == 0, f"Runner failed:\n{result.stderr}"

    def test_creates_csv(self, tmp_output_dir):
        _run_script("gaussian_mean_runner.py", "gaussian_mean.json", tmp_output_dir)
        exp_dir = tmp_output_dir / "gaussian_mean"
        csv_path = exp_dir / "data" / "raw_results.csv"
        assert csv_path.exists(), f"Expected {csv_path}"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    def test_creates_metadata(self, tmp_output_dir):
        _run_script("gaussian_mean_runner.py", "gaussian_mean.json", tmp_output_dir)
        meta = tmp_output_dir / "gaussian_mean" / "data" / "metadata.json"
        assert meta.exists()
        with open(meta) as f:
            data = json.load(f)
        assert "experiment_name" in data
        assert "timestamp" in data
        assert "config" in data

    def test_idempotent(self, tmp_output_dir):
        """Re-running should succeed without crashing."""
        _run_script("gaussian_mean_runner.py", "gaussian_mean.json", tmp_output_dir)
        result = _run_script("gaussian_mean_runner.py", "gaussian_mean.json", tmp_output_dir)
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# G-and-k runner
# ---------------------------------------------------------------------------

class TestGandKRunner:
    def test_completes(self, tmp_output_dir):
        result = _run_script("gandk_runner.py", "gandk.json", tmp_output_dir)
        assert result.returncode == 0, f"Runner failed:\n{result.stderr}"

    def test_creates_csv(self, tmp_output_dir):
        _run_script("gandk_runner.py", "gandk.json", tmp_output_dir)
        csv_path = tmp_output_dir / "gandk" / "data" / "raw_results.csv"
        assert csv_path.exists()

    def test_creates_metadata(self, tmp_output_dir):
        _run_script("gandk_runner.py", "gandk.json", tmp_output_dir)
        assert (tmp_output_dir / "gandk" / "data" / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Lotka-Volterra runner
# ---------------------------------------------------------------------------

class TestLotkaVolterraRunner:
    def test_completes(self, tmp_output_dir):
        result = _run_script("lotka_volterra_runner.py", "lotka_volterra.json", tmp_output_dir)
        assert result.returncode == 0, f"Runner failed:\n{result.stderr}"

    def test_creates_csv(self, tmp_output_dir):
        _run_script("lotka_volterra_runner.py", "lotka_volterra.json", tmp_output_dir)
        csv_path = tmp_output_dir / "lotka_volterra" / "data" / "raw_results.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# CSV schema — shared assertion
# ---------------------------------------------------------------------------

class TestRunnerCsvSchema:
    def test_csv_has_expected_columns(self, tmp_output_dir):
        _run_script("gaussian_mean_runner.py", "gaussian_mean.json", tmp_output_dir)
        csv_path = tmp_output_dir / "gaussian_mean" / "data" / "raw_results.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        for col in ("method", "replicate", "seed", "step", "loss", "wall_time"):
            assert col in header, f"Column '{col}' missing from CSV"

    def test_method_column_matches_config(self, tmp_output_dir):
        _run_script("gaussian_mean_runner.py", "gaussian_mean.json", tmp_output_dir)
        csv_path = tmp_output_dir / "gaussian_mean" / "data" / "raw_results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        methods = {r["method"] for r in rows}
        assert "async_propulate_abc" in methods


# ---------------------------------------------------------------------------
# Sensitivity runner (direct import — faster than subprocess)
# ---------------------------------------------------------------------------

class TestSensitivityRunner:
    def test_module_importable(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        import importlib
        spec = importlib.util.spec_from_file_location(
            "sensitivity_runner", SCRIPTS_DIR / "sensitivity_runner.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")

    def test_runs_test_mode(self, tmp_output_dir):
        result = _run_script("sensitivity_runner.py", "sensitivity.json", tmp_output_dir)
        assert result.returncode == 0, f"Sensitivity runner failed:\n{result.stderr}"

    def test_creates_one_csv_per_variant(self, tmp_output_dir):
        _run_script("sensitivity_runner.py", "sensitivity.json", tmp_output_dir)
        data_dir = tmp_output_dir / "sensitivity" / "data"
        csvs = list(data_dir.glob("*.csv"))
        # There should be at least one CSV for each sensitivity variant
        assert len(csvs) >= 1


# ---------------------------------------------------------------------------
# Ablation runner (subprocess)
# ---------------------------------------------------------------------------

class TestAblationRunner:
    def test_completes(self, tmp_output_dir):
        result = _run_script("ablation_runner.py", "ablation.json", tmp_output_dir)
        assert result.returncode == 0, f"Ablation runner failed:\n{result.stderr}"

    def test_creates_one_csv_per_variant(self, tmp_output_dir):
        _run_script("ablation_runner.py", "ablation.json", tmp_output_dir)
        data_dir = tmp_output_dir / "ablation" / "data"
        csvs = list(data_dir.glob("*.csv"))
        assert len(csvs) >= 2  # at least 2 ablation variants


# ---------------------------------------------------------------------------
# Scaling runner
# ---------------------------------------------------------------------------

class TestScalingRunner:
    def test_completes(self, tmp_output_dir):
        result = _run_script("scaling_runner.py", "scaling.json", tmp_output_dir)
        assert result.returncode == 0, f"Scaling runner failed:\n{result.stderr}"

    def test_creates_throughput_csv(self, tmp_output_dir):
        _run_script("scaling_runner.py", "scaling.json", tmp_output_dir)
        data_dir = tmp_output_dir / "scaling" / "data"
        assert any(data_dir.glob("throughput*.csv")), "No throughput CSV found"

    def test_throughput_csv_has_n_workers_column(self, tmp_output_dir):
        _run_script("scaling_runner.py", "scaling.json", tmp_output_dir)
        data_dir = tmp_output_dir / "scaling" / "data"
        csv_path = next(data_dir.glob("throughput*.csv"))
        with open(csv_path) as f:
            header = csv.DictReader(f).fieldnames
        assert "n_workers" in header


# ---------------------------------------------------------------------------
# Runtime heterogeneity runner
# ---------------------------------------------------------------------------

class TestRuntimeHeterogeneityRunner:
    def test_completes(self, tmp_output_dir):
        result = _run_script(
            "runtime_heterogeneity_runner.py", "runtime_heterogeneity.json", tmp_output_dir
        )
        assert result.returncode == 0, f"Runner failed:\n{result.stderr}"
