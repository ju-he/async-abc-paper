"""Tests for experiment runner scripts."""
import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers


def _rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


class TestRunnerImports:
    def test_importing_sensitivity_runner_does_not_load_convergence(self):
        sys.modules.pop("async_abc.analysis.convergence", None)
        sys.modules.pop("ot", None)

        module = test_helpers.import_runner_module("sensitivity_runner.py")

        assert hasattr(module, "main")
        assert "async_abc.analysis.convergence" not in sys.modules
        assert "ot" not in sys.modules


class TestRunnerCliSmoke:
    def test_gaussian_runner_cli_smoke(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 80, "k": 20},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={
                "posterior": True,
                "archive_evolution": False,
                "corner": True,
                "tolerance_trajectory": True,
                "quality_vs_time": True,
            },
        )
        config_path = test_helpers.write_config(tmp_path, "gaussian_cli.json", cfg)

        result = test_helpers.run_runner_subprocess(
            "gaussian_mean_runner.py",
            config_path,
            tmp_path,
        )

        assert result.returncode == 0, result.stderr
        assert (tmp_path / "gaussian_mean" / "data" / "raw_results.csv").exists()

    def test_sensitivity_runner_cli_smoke(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "sensitivity.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 60, "k": 10},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"sensitivity_heatmap": False},
            replace_top_level={
                "sensitivity_grid": {
                    "k": [10],
                    "perturbation_scale": [0.8],
                    "scheduler_type": ["acceptance_rate"],
                    "tol_init_multiplier": [1.0],
                }
            },
        )
        config_path = test_helpers.write_config(tmp_path, "sensitivity_cli.json", cfg)

        result = test_helpers.run_runner_subprocess(
            "sensitivity_runner.py",
            config_path,
            tmp_path,
        )

        assert result.returncode == 0, result.stderr
        assert list((tmp_path / "sensitivity" / "data").glob("sensitivity_*.csv"))


class TestGaussianMeanRunner:
    def test_creates_csv(self, gaussian_runner_artifact):
        csv_path = gaussian_runner_artifact["root"] / "gaussian_mean" / "data" / "raw_results.csv"
        rows = _rows(csv_path)
        assert rows

    def test_creates_metadata(self, gaussian_runner_artifact):
        meta = gaussian_runner_artifact["root"] / "gaussian_mean" / "data" / "metadata.json"
        data = json.loads(meta.read_text())
        assert "experiment_name" in data
        assert "timestamp" in data
        assert "config" in data

    def test_rerun_is_idempotent(self, gaussian_runner_artifact, tmp_path):
        test_helpers.copy_output_tree(gaussian_runner_artifact["root"], tmp_path)
        config_path = test_helpers.clone_artifact_config(gaussian_runner_artifact, tmp_path)

        test_helpers.run_runner_main("gaussian_mean_runner.py", config_path, tmp_path)

        csv_path = tmp_path / "gaussian_mean" / "data" / "raw_results.csv"
        assert csv_path.exists()

    def test_creates_phase3_plots(self, gaussian_runner_artifact):
        plots_dir = gaussian_runner_artifact["root"] / "gaussian_mean" / "plots"
        assert (plots_dir / "quality_vs_time.pdf").exists()
        assert (plots_dir / "tolerance_trajectory.pdf").exists()
        assert (plots_dir / "corner.pdf").exists()


class TestBenchmarkRunnerArtifacts:
    def test_gandk_creates_csv_and_metadata(self, gandk_runner_artifact):
        root = gandk_runner_artifact["root"] / "gandk"
        assert (root / "data" / "raw_results.csv").exists()
        assert (root / "data" / "metadata.json").exists()

    def test_lotka_volterra_creates_csv(self, lotka_runner_artifact):
        csv_path = lotka_runner_artifact["root"] / "lotka_volterra" / "data" / "raw_results.csv"
        assert csv_path.exists()


class TestRunnerCsvSchema:
    def test_csv_has_expected_columns(self, gaussian_runner_artifact):
        csv_path = gaussian_runner_artifact["root"] / "gaussian_mean" / "data" / "raw_results.csv"
        with open(csv_path) as f:
            header = csv.DictReader(f).fieldnames
        for col in ("method", "replicate", "seed", "step", "loss", "wall_time"):
            assert col in header

    def test_method_column_matches_config(self, gaussian_runner_artifact):
        csv_path = gaussian_runner_artifact["root"] / "gaussian_mean" / "data" / "raw_results.csv"
        methods = {row["method"] for row in _rows(csv_path)}
        assert methods == {"rejection_abc"}


class TestSensitivityRunner:
    def test_creates_one_csv_per_variant(self, sensitivity_runner_artifact):
        data_dir = sensitivity_runner_artifact["root"] / "sensitivity" / "data"
        csvs = list(data_dir.glob("sensitivity_*.csv"))
        assert len(csvs) == 1

    def test_applies_tol_init_multiplier(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "sensitivity.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 80, "k": 10, "tol_init": 5.0},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"sensitivity_heatmap": False},
            replace_top_level={"sensitivity_grid": {"tol_init_multiplier": [0.5, 2.0]}},
        )
        config_path = test_helpers.write_config(tmp_path, "sensitivity_tol.json", cfg)

        test_helpers.run_runner_main("sensitivity_runner.py", config_path, tmp_path)

        csvs = sorted((tmp_path / "sensitivity" / "data").glob("sensitivity_*.csv"))
        assert len(csvs) == 2
        methods = set()
        tolerance_by_name = {}
        for csv_path in csvs:
            rows = _rows(csv_path)
            methods.update(row["method"] for row in rows)
            tolerance_by_name[csv_path.stem] = {
                float(row["tolerance"]) for row in rows if row.get("tolerance")
            }
        assert any("tol_init_multiplier=0.5" in method for method in methods)
        assert any("tol_init_multiplier=2.0" in method for method in methods)
        assert tolerance_by_name["sensitivity_tol_init_multiplier=0.5"] == {2.5}
        assert tolerance_by_name["sensitivity_tol_init_multiplier=2.0"] == {10.0}


class TestAblationRunner:
    def test_creates_one_csv_per_variant(self, ablation_runner_artifact):
        data_dir = ablation_runner_artifact["root"] / "ablation" / "data"
        csvs = list(data_dir.glob("ablation_*.csv"))
        assert len(csvs) == 2


class TestScalingRunner:
    def test_creates_throughput_csv(self, scaling_runner_artifact):
        data_dir = scaling_runner_artifact["root"] / "scaling" / "data"
        assert any(data_dir.glob("throughput*.csv"))

    def test_throughput_csv_has_expected_worker_counts(self, scaling_runner_artifact):
        csv_path = scaling_runner_artifact["root"] / "scaling" / "data" / "throughput_summary.csv"
        rows = _rows(csv_path)
        assert {int(row["n_workers"]) for row in rows} == {1, 4}


class TestRuntimeHeterogeneityRunner:
    def test_creates_gantt_plot(self, runtime_heterogeneity_runner_artifact):
        plots_dir = runtime_heterogeneity_runner_artifact["root"] / "runtime_heterogeneity" / "plots"
        assert (plots_dir / "worker_gantt.pdf").exists()


class TestStragglerRunner:
    def test_tags_records_with_slowdown(self, straggler_runner_artifact):
        csv_path = straggler_runner_artifact["root"] / "straggler" / "data" / "raw_results.csv"
        methods = {row["method"] for row in _rows(csv_path)}
        assert any("straggler_slowdown" in method for method in methods)
        assert any("1x" in method for method in methods)
        assert any("5x" in method for method in methods)

    def test_throughput_csv_has_no_inf_values(self, straggler_runner_artifact):
        csv_path = (
            straggler_runner_artifact["root"]
            / "straggler"
            / "data"
            / "throughput_vs_slowdown_summary.csv"
        )
        for row in _rows(csv_path):
            val = row.get("throughput_sims_per_s", "")
            if val:
                assert not math.isinf(float(val))
