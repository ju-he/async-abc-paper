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


def _completed_run(script_name: str, config_name: str, output_dir: Path) -> dict:
    result = _run_script(script_name, config_name, output_dir)
    assert result.returncode == 0, f"Runner failed:\n{result.stderr}"
    return {
        "result": result,
        "output_dir": output_dir,
    }


@pytest.fixture(scope="module")
def gaussian_mean_run(tmp_path_factory):
    return _completed_run(
        "gaussian_mean_runner.py",
        "gaussian_mean.json",
        tmp_path_factory.mktemp("gaussian_mean_runner"),
    )


@pytest.fixture(scope="module")
def gandk_run(tmp_path_factory):
    return _completed_run(
        "gandk_runner.py",
        "gandk.json",
        tmp_path_factory.mktemp("gandk_runner"),
    )


@pytest.fixture(scope="module")
def lotka_volterra_run(tmp_path_factory):
    return _completed_run(
        "lotka_volterra_runner.py",
        "lotka_volterra.json",
        tmp_path_factory.mktemp("lotka_runner"),
    )


@pytest.fixture(scope="module")
def sensitivity_run(tmp_path_factory):
    return _completed_run(
        "sensitivity_runner.py",
        "sensitivity.json",
        tmp_path_factory.mktemp("sensitivity_runner"),
    )


@pytest.fixture(scope="module")
def ablation_run(tmp_path_factory):
    return _completed_run(
        "ablation_runner.py",
        "ablation.json",
        tmp_path_factory.mktemp("ablation_runner"),
    )


@pytest.fixture(scope="module")
def scaling_run(tmp_path_factory):
    return _completed_run(
        "scaling_runner.py",
        "scaling.json",
        tmp_path_factory.mktemp("scaling_runner"),
    )


@pytest.fixture(scope="module")
def runtime_heterogeneity_run(tmp_path_factory):
    return _completed_run(
        "runtime_heterogeneity_runner.py",
        "runtime_heterogeneity.json",
        tmp_path_factory.mktemp("runtime_heterogeneity_runner"),
    )


@pytest.fixture(scope="module")
def straggler_run(tmp_path_factory):
    cfg = {
        "experiment_name": "straggler",
        "benchmark": {
            "name": "gaussian_mean",
            "observed_data_seed": 42,
            "n_obs": 40,
            "true_mu": 0.0,
            "sigma_obs": 1.0,
            "prior_low": -5.0,
            "prior_high": 5.0,
        },
        "methods": ["async_propulate_abc", "abc_smc_baseline"],
        "inference": {
            "max_simulations": 200,
            "n_workers": 2,
            "k": 20,
            "tol_init": 5.0,
            "n_generations": 2,
            "scheduler_type": "acceptance_rate",
            "perturbation_scale": 0.8,
        },
        "execution": {
            "n_replicates": 1,
            "base_seed": 0,
        },
        "straggler": {
            "straggler_rank": 0,
            "base_sleep_s": 0.1,
            "slowdown_factor": [1, 5],
        },
        "plots": {
            "throughput_vs_slowdown": True,
            "gantt": True,
        },
    }
    root = tmp_path_factory.mktemp("straggler_runner")
    config_path = root / "straggler.json"
    config_path.write_text(json.dumps(cfg))
    result = subprocess.run(
        [
            PYTHON,
            str(SCRIPTS_DIR / "straggler_runner.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(root),
            "--test",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Runner failed:\n{result.stderr}"
    return {
        "result": result,
        "output_dir": root,
    }


# ---------------------------------------------------------------------------
# Gaussian mean runner
# ---------------------------------------------------------------------------

class TestGaussianMeanRunner:
    def test_completes(self, gaussian_mean_run):
        assert gaussian_mean_run["result"].returncode == 0

    def test_creates_csv(self, gaussian_mean_run):
        exp_dir = gaussian_mean_run["output_dir"] / "gaussian_mean"
        csv_path = exp_dir / "data" / "raw_results.csv"
        assert csv_path.exists(), f"Expected {csv_path}"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    def test_creates_metadata(self, gaussian_mean_run):
        meta = gaussian_mean_run["output_dir"] / "gaussian_mean" / "data" / "metadata.json"
        assert meta.exists()
        with open(meta) as f:
            data = json.load(f)
        assert "experiment_name" in data
        assert "timestamp" in data
        assert "config" in data

    def test_idempotent(self, gaussian_mean_run):
        """Re-running should succeed without crashing."""
        result = _run_script(
            "gaussian_mean_runner.py",
            "gaussian_mean.json",
            gaussian_mean_run["output_dir"],
        )
        assert result.returncode == 0

    def test_creates_phase3_plots(self, gaussian_mean_run):
        plots_dir = gaussian_mean_run["output_dir"] / "gaussian_mean" / "plots"
        assert (plots_dir / "quality_vs_time.pdf").exists()
        assert (plots_dir / "tolerance_trajectory.pdf").exists()
        assert (plots_dir / "corner.pdf").exists()


# ---------------------------------------------------------------------------
# G-and-k runner
# ---------------------------------------------------------------------------

class TestGandKRunner:
    def test_completes(self, gandk_run):
        assert gandk_run["result"].returncode == 0

    def test_creates_csv(self, gandk_run):
        csv_path = gandk_run["output_dir"] / "gandk" / "data" / "raw_results.csv"
        assert csv_path.exists()

    def test_creates_metadata(self, gandk_run):
        assert (gandk_run["output_dir"] / "gandk" / "data" / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Lotka-Volterra runner
# ---------------------------------------------------------------------------

class TestLotkaVolterraRunner:
    def test_completes(self, lotka_volterra_run):
        assert lotka_volterra_run["result"].returncode == 0

    def test_creates_csv(self, lotka_volterra_run):
        csv_path = lotka_volterra_run["output_dir"] / "lotka_volterra" / "data" / "raw_results.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# CSV schema — shared assertion
# ---------------------------------------------------------------------------

class TestRunnerCsvSchema:
    def test_csv_has_expected_columns(self, gaussian_mean_run):
        csv_path = gaussian_mean_run["output_dir"] / "gaussian_mean" / "data" / "raw_results.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        for col in ("method", "replicate", "seed", "step", "loss", "wall_time"):
            assert col in header, f"Column '{col}' missing from CSV"

    def test_method_column_matches_config(self, gaussian_mean_run):
        csv_path = gaussian_mean_run["output_dir"] / "gaussian_mean" / "data" / "raw_results.csv"
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

    def test_runs_test_mode(self, sensitivity_run):
        assert sensitivity_run["result"].returncode == 0

    def test_creates_one_csv_per_variant(self, sensitivity_run):
        data_dir = sensitivity_run["output_dir"] / "sensitivity" / "data"
        csvs = list(data_dir.glob("*.csv"))
        # There should be at least one CSV for each sensitivity variant
        assert len(csvs) >= 1

    def test_sensitivity_runner_applies_tol_init_multiplier(self, tmp_path):
        cfg = json.loads((CONFIGS_DIR / "sensitivity.json").read_text())
        cfg["sensitivity_grid"] = {
            "tol_init_multiplier": [0.5, 2.0],
        }
        cfg["methods"] = ["rejection_abc"]
        cfg["execution"]["n_replicates"] = 1
        cfg["inference"]["max_simulations"] = 300
        cfg_path = tmp_path / "sensitivity_tol.json"
        cfg_path.write_text(json.dumps(cfg))

        result = subprocess.run(
            [
                PYTHON,
                str(SCRIPTS_DIR / "sensitivity_runner.py"),
                "--config",
                str(cfg_path),
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"Sensitivity runner failed:\n{result.stderr}"

        data_dir = tmp_path / "sensitivity" / "data"
        csvs = sorted(data_dir.glob("sensitivity_*.csv"))
        names = [path.stem for path in csvs]
        assert any("tol_init_multiplier=0.5" in name for name in names)
        assert any("tol_init_multiplier=2.0" in name for name in names)

        methods = set()
        tolerance_by_name = {}
        for csv_path in csvs:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            methods.update(row["method"] for row in rows)
            tolerances = {float(row["tolerance"]) for row in rows if row.get("tolerance")}
            tolerance_by_name[csv_path.stem] = tolerances
        assert any("tol_init_multiplier=0.5" in method for method in methods)
        assert any("tol_init_multiplier=2.0" in method for method in methods)
        assert tolerance_by_name["sensitivity_tol_init_multiplier=0.5"] == {2.5}
        assert tolerance_by_name["sensitivity_tol_init_multiplier=2.0"] == {10.0}


# ---------------------------------------------------------------------------
# Ablation runner (subprocess)
# ---------------------------------------------------------------------------

class TestAblationRunner:
    def test_completes(self, ablation_run):
        assert ablation_run["result"].returncode == 0

    def test_creates_one_csv_per_variant(self, ablation_run):
        data_dir = ablation_run["output_dir"] / "ablation" / "data"
        csvs = list(data_dir.glob("*.csv"))
        assert len(csvs) >= 2  # at least 2 ablation variants


# ---------------------------------------------------------------------------
# Scaling runner
# ---------------------------------------------------------------------------

class TestScalingRunner:
    def test_completes(self, scaling_run):
        assert scaling_run["result"].returncode == 0

    def test_creates_throughput_csv(self, scaling_run):
        data_dir = scaling_run["output_dir"] / "scaling" / "data"
        assert any(data_dir.glob("throughput*.csv")), "No throughput CSV found"

    def test_throughput_csv_has_n_workers_column(self, scaling_run):
        data_dir = scaling_run["output_dir"] / "scaling" / "data"
        csv_path = next(data_dir.glob("throughput*.csv"))
        with open(csv_path) as f:
            header = csv.DictReader(f).fieldnames
        assert "n_workers" in header


# ---------------------------------------------------------------------------
# Runtime heterogeneity runner
# ---------------------------------------------------------------------------

class TestRuntimeHeterogeneityRunner:
    def test_completes(self, runtime_heterogeneity_run):
        assert runtime_heterogeneity_run["result"].returncode == 0

    def test_creates_gantt_plot(self, runtime_heterogeneity_run):
        plots_dir = runtime_heterogeneity_run["output_dir"] / "runtime_heterogeneity" / "plots"
        assert (plots_dir / "worker_gantt.pdf").exists()


# ---------------------------------------------------------------------------
# Straggler runner
# ---------------------------------------------------------------------------

class TestStragglerRunner:
    def test_straggler_runner_test_mode(self, straggler_run):
        csv_path = straggler_run["output_dir"] / "straggler" / "data" / "raw_results.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        methods = {r["method"] for r in rows}
        assert any("straggler_slowdown" in method for method in methods)

    def test_straggler_runner_tags_records(self, straggler_run):
        csv_path = straggler_run["output_dir"] / "straggler" / "data" / "raw_results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        methods = {r["method"] for r in rows}
        assert any("1x" in method for method in methods)
        assert any("5x" in method for method in methods)

    def test_throughput_csv_has_no_inf_values(self, straggler_run):
        import math
        throughput_path = (
            straggler_run["output_dir"] / "straggler" / "data" / "throughput_vs_slowdown_summary.csv"
        )
        assert throughput_path.exists(), "throughput_vs_slowdown_summary.csv was not created"
        with open(throughput_path) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            val = row.get("throughput_sims_per_s", "")
            if val:
                assert not math.isinf(float(val)), (
                    f"throughput_sims_per_s contains inf in row: {row}"
                )
