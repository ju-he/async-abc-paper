"""Tests for experiment runner scripts."""
import copy
import csv
import json
import math
import sys
import types
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

    def test_cellular_potts_runner_routes_scratch_output_under_run_dir(self, tmp_path, monkeypatch):
        module = test_helpers.import_runner_module("cellular_potts_runner.py")
        captured = {}

        cfg = test_helpers.make_fast_runner_config(
            "cellular_potts.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 5, "k": 2},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={},
        )

        monkeypatch.setattr(module, "configure_logging", lambda: None)
        monkeypatch.setattr(module, "load_config", lambda path, test_mode=False, small_mode=False: cfg)
        monkeypatch.setattr(module, "is_root_rank", lambda: True)
        monkeypatch.setattr(module, "compute_corrected_estimate", lambda *args, **kwargs: 0.0)
        monkeypatch.setattr(module, "write_timing_csv", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "write_metadata", lambda *args, **kwargs: None)

        def _fake_run_experiment(runtime_cfg, output_dir):
            captured["cfg"] = runtime_cfg
            captured["output_dir"] = output_dir
            return []

        monkeypatch.setattr(module, "run_experiment", _fake_run_experiment)

        module.main(
            [
                "--config",
                str(tmp_path / "cellular_potts.json"),
                "--output-dir",
                str(tmp_path),
            ]
        )

        assert captured["cfg"]["benchmark"]["output_dir"] == str(
            tmp_path / "cellular_potts" / "cpm_sims"
        )


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
        assert data["experiment_role"] == "validity"
        assert data["stop_policy"] == "fixed_budget"
        assert data["method_comparison_roles"]["rejection_abc"] == "small_model_reference"

    def test_rerun_is_idempotent(self, gaussian_runner_artifact, tmp_path):
        test_helpers.copy_output_tree(gaussian_runner_artifact["root"], tmp_path)
        config_path = test_helpers.clone_artifact_config(gaussian_runner_artifact, tmp_path)

        test_helpers.run_runner_main("gaussian_mean_runner.py", config_path, tmp_path)

        csv_path = tmp_path / "gaussian_mean" / "data" / "raw_results.csv"
        assert csv_path.exists()

    def test_creates_phase3_plots(self, gaussian_runner_artifact):
        plots_dir = gaussian_runner_artifact["root"] / "gaussian_mean" / "plots"
        data_dir = gaussian_runner_artifact["root"] / "gaussian_mean" / "data"
        assert (plots_dir / "progress_summary.pdf").exists()
        assert (plots_dir / "progress_diagnostic.pdf").exists()
        assert (plots_dir / "quality_vs_wall_time.pdf").exists()
        assert (plots_dir / "quality_vs_wall_time_diagnostic.pdf").exists()
        assert (plots_dir / "quality_vs_posterior_samples.pdf").exists()
        assert (plots_dir / "quality_vs_posterior_samples_diagnostic.pdf").exists()
        assert (plots_dir / "quality_vs_attempt_budget.pdf").exists()
        assert (plots_dir / "quality_vs_attempt_budget_diagnostic.pdf").exists()
        assert (plots_dir / "time_to_target_summary_meta.json").exists()
        assert (plots_dir / "time_to_target_diagnostic_meta.json").exists()
        assert (plots_dir / "tolerance_trajectory.pdf").exists()
        assert (plots_dir / "tolerance_trajectory_diagnostic.pdf").exists()
        assert (plots_dir / "corner.pdf").exists()
        assert (data_dir / "plot_audit.csv").exists()

    def test_small_mode_writes_run_mode_metadata_and_timing(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 80, "k": 20},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={},
        )
        small_cfg = copy.deepcopy(cfg)
        small_cfg["inference"]["max_simulations"] = 40
        config_path = test_helpers.write_config(tmp_path, "gaussian_small.json", cfg)
        small_dir = tmp_path / "small"
        small_dir.mkdir()
        test_helpers.write_config(small_dir, config_path.name, small_cfg)

        test_helpers.run_runner_main(
            "gaussian_mean_runner.py",
            config_path,
            tmp_path,
            small_mode=True,
        )

        metadata = json.loads((tmp_path / "gaussian_mean" / "data" / "metadata.json").read_text())
        timing_rows = _rows(tmp_path / "gaussian_mean" / "data" / "timing.csv")

        assert metadata["run_mode"] == "small"
        assert metadata["test_mode"] is False
        assert metadata["config"]["execution"]["config_tier"] == "small"
        assert timing_rows[-1]["run_mode"] == "small"
        assert timing_rows[-1]["test_mode"] == "False"


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

    def test_sensitivity_gandk_uses_sensitivity_shard_finalizer(self):
        from async_abc.utils.shard_finalizers import (
            _FINALIZER_REGISTRY,
            finalize_sensitivity_experiment,
        )

        assert _FINALIZER_REGISTRY["sensitivity_gandk"] is finalize_sensitivity_experiment

    def test_small_mode_uses_small_grid_without_test_collapse(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "sensitivity.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 40, "k": 10, "n_workers": 1},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"sensitivity_heatmap": False},
            replace_top_level={
                "sensitivity_grid": {
                    "k": [10, 20, 30],
                    "perturbation_scale": [0.4, 0.8, 1.2],
                }
            },
        )
        small_cfg = copy.deepcopy(cfg)
        small_cfg["sensitivity_grid"] = {
            "k": [10, 20],
            "perturbation_scale": [0.4, 0.8],
        }
        config_path = test_helpers.write_config(tmp_path, "sensitivity_small.json", cfg)
        small_dir = tmp_path / "small"
        small_dir.mkdir()
        test_helpers.write_config(small_dir, config_path.name, small_cfg)

        test_helpers.run_runner_main(
            "sensitivity_runner.py",
            config_path,
            tmp_path / "small_only",
            small_mode=True,
        )
        test_helpers.run_runner_main(
            "sensitivity_runner.py",
            config_path,
            tmp_path / "small_test",
            small_mode=True,
            test_mode=True,
        )

        small_csvs = list((tmp_path / "small_only" / "sensitivity" / "data").glob("sensitivity_*.csv"))
        small_test_csvs = list((tmp_path / "small_test" / "sensitivity" / "data").glob("sensitivity_*.csv"))

        assert len(small_csvs) == 4
        assert len(small_test_csvs) == 1


class TestAblationRunner:
    def test_creates_one_csv_per_variant(self, ablation_runner_artifact):
        data_dir = ablation_runner_artifact["root"] / "ablation" / "data"
        csvs = list(data_dir.glob("ablation_*.csv"))
        assert len(csvs) == 2


class TestScalingRunner:
    def test_creates_summary_csvs(self, scaling_runner_artifact):
        data_dir = scaling_runner_artifact["root"] / "scaling" / "data"
        assert (data_dir / "throughput_summary.csv").exists()
        assert (data_dir / "budget_summary.csv").exists()
        assert (data_dir / "raw_results.csv").exists()

    def test_throughput_csv_has_expected_worker_counts_and_k_values(self, scaling_runner_artifact):
        csv_path = scaling_runner_artifact["root"] / "scaling" / "data" / "throughput_summary.csv"
        rows = _rows(csv_path)
        assert {int(row["n_workers"]) for row in rows} == {1, 4}
        assert {int(row["k"]) for row in rows} == {10, 50}
        assert {row["base_method"] for row in rows} == {"rejection_abc"}
        assert {row["stop_policy"] for row in rows} == {"simulation_cap_approx"}

    def test_budget_summary_has_expected_budget_rows(self, scaling_runner_artifact):
        csv_path = scaling_runner_artifact["root"] / "scaling" / "data" / "budget_summary.csv"
        rows = _rows(csv_path)
        assert {float(row["budget_s"]) for row in rows} == {0.05, 0.1}
        assert {int(row["n_workers"]) for row in rows} == {1, 4}
        assert {int(row["k"]) for row in rows} == {10, 50}

    def test_requested_max_simulations_uses_policy(self):
        module = test_helpers.import_runner_module("scaling_runner.py")

        requested = module._requested_max_simulations(
            {"max_simulations": 80},
            n_workers=4,
            k=50,
            policy={"min_total": 40, "per_worker": 30, "k_factor": 2},
        )

        assert requested == 120

    def test_simulation_count_uses_attempts_before_raw_record_count(self):
        module = test_helpers.import_runner_module("scaling_runner.py")

        attempt_records = [
            test_helpers.ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=1,
                step=1,
                params={"mu": 0.0},
                loss=0.1,
                wall_time=0.1,
                record_kind="simulation_attempt",
                attempt_count=1,
            ),
            test_helpers.ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=1,
                step=2,
                params={"mu": 0.1},
                loss=0.2,
                wall_time=0.2,
                record_kind="simulation_attempt",
                attempt_count=2,
            ),
            test_helpers.ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=1,
                step=1,
                params={"mu": 0.1},
                loss=0.2,
                wall_time=0.2,
                record_kind="population_particle",
                attempt_count=2,
            ),
        ]
        accepted_only_records = [
            test_helpers.ParticleRecord(
                method="rejection_abc",
                replicate=0,
                seed=1,
                step=1,
                params={"mu": 0.0},
                loss=0.1,
                wall_time=0.1,
                record_kind="accepted_particle",
                attempt_count=4,
            ),
            test_helpers.ParticleRecord(
                method="rejection_abc",
                replicate=0,
                seed=1,
                step=2,
                params={"mu": 0.2},
                loss=0.2,
                wall_time=0.2,
                record_kind="accepted_particle",
                attempt_count=9,
            ),
        ]

        assert module._simulation_count(attempt_records) == 2
        assert module._simulation_count(accepted_only_records) == 9

    def test_rebuild_scaling_outputs_merges_worker_shards(self, tmp_path, monkeypatch):
        module = test_helpers.import_runner_module("scaling_runner.py")
        output_dir = module.OutputDir(tmp_path, "scaling").ensure()
        rows = [
            {
                "base_method": "async_propulate_abc",
                "method_variant": "async_propulate_abc__k50__w4",
                "stop_policy": "simulation_cap_approx",
                "k": 50,
                "n_workers": 4,
                "replicate": 0,
                "seed": 2,
                "requested_max_simulations": 120,
                "max_wall_time_s": 0.1,
                "elapsed_wall_time_s": 2.0,
                "n_simulations": 20,
                "throughput_sims_per_s": 10.0,
                "final_quality_wasserstein": 1.0,
                "final_n_particles": 10,
                "final_tolerance": 0.5,
                "state_kind": "accepted_prefix",
                "test_mode": False,
            },
            {
                "base_method": "async_propulate_abc",
                "method_variant": "async_propulate_abc__k10__w1",
                "stop_policy": "simulation_cap_approx",
                "k": 10,
                "n_workers": 1,
                "replicate": 0,
                "seed": 1,
                "requested_max_simulations": 80,
                "max_wall_time_s": 0.1,
                "elapsed_wall_time_s": 2.0,
                "n_simulations": 10,
                "throughput_sims_per_s": 5.0,
                "final_quality_wasserstein": 2.0,
                "final_n_particles": 5,
                "final_tolerance": 1.0,
                "state_kind": "accepted_prefix",
                "test_mode": False,
            },
        ]
        budget_rows = [
            {
                "base_method": "async_propulate_abc",
                "method_variant": "async_propulate_abc__k10__w1",
                "k": 10,
                "n_workers": 1,
                "replicate": 0,
                "seed": 1,
                "budget_s": 0.1,
                "requested_max_simulations": 80,
                "max_wall_time_s": 0.1,
                "elapsed_wall_time_s": 0.1,
                "attempts_by_budget": 8,
                "posterior_samples_by_budget": 4,
                "quality_wasserstein_by_budget": 2.0,
                "best_tolerance_by_budget": 1.0,
                "test_mode": False,
            },
            {
                "base_method": "async_propulate_abc",
                "method_variant": "async_propulate_abc__k50__w4",
                "k": 50,
                "n_workers": 4,
                "replicate": 0,
                "seed": 2,
                "budget_s": 0.1,
                "requested_max_simulations": 120,
                "max_wall_time_s": 0.1,
                "elapsed_wall_time_s": 0.1,
                "attempts_by_budget": 18,
                "posterior_samples_by_budget": 9,
                "quality_wasserstein_by_budget": 1.0,
                "best_tolerance_by_budget": 0.5,
                "test_mode": False,
            },
        ]
        captured = {}
        monkeypatch.setattr(
            module,
            "write_metadata",
            lambda _output_dir, _cfg, extra=None: captured.setdefault("worker_counts", extra["worker_counts"]),
        )

        module._write_rows_atomic(
            module._throughput_shard_path(output_dir.data, 1, 10),
            [rows[1]],
            module._THROUGHPUT_FIELDNAMES,
        )
        module._write_rows_atomic(
            module._throughput_shard_path(output_dir.data, 4, 50),
            [rows[0]],
            module._THROUGHPUT_FIELDNAMES,
        )
        module._write_rows_atomic(
            module._budget_shard_path(output_dir.data, 1, 10),
            [budget_rows[0]],
            module._BUDGET_FIELDNAMES,
        )
        module._write_rows_atomic(
            module._budget_shard_path(output_dir.data, 4, 50),
            [budget_rows[1]],
            module._BUDGET_FIELDNAMES,
        )
        aggregate_rows = module.rebuild_scaling_outputs(
            output_dir,
            {"plots": {}, "experiment_name": "scaling", "scaling": {"wall_time_budgets_s": [0.1]}},
        )

        csv_rows = _rows(output_dir.data / "throughput_summary.csv")
        assert [int(row["n_workers"]) for row in csv_rows] == [1, 4]
        assert [int(row["k"]) for row in csv_rows] == [10, 50]
        assert [int(row["n_workers"]) for row in aggregate_rows] == [1, 4]
        assert captured["worker_counts"] == [1, 4]

    def test_rebuild_scaling_metadata_includes_stop_policy_mapping(self, tmp_path, monkeypatch):
        module = test_helpers.import_runner_module("scaling_runner.py")
        output_dir = module.OutputDir(tmp_path, "scaling").ensure()
        captured = {}
        monkeypatch.setattr(
            module,
            "write_metadata",
            lambda _output_dir, _cfg, extra=None: captured.update(extra or {}),
        )
        module.rebuild_scaling_outputs(
            output_dir,
            {
                "experiment_name": "scaling",
                "methods": ["async_propulate_abc", "abc_smc_baseline"],
                "plots": {},
                "scaling": {"wall_time_budgets_s": [0.1]},
            },
            fallback_rows=[
                {
                    "base_method": "async_propulate_abc",
                    "method_variant": "async_propulate_abc__k10__w1",
                    "stop_policy": "simulation_cap_approx",
                    "k": 10,
                    "n_workers": 1,
                    "replicate": 0,
                    "seed": 1,
                    "requested_max_simulations": 60,
                    "max_wall_time_s": 0.1,
                    "elapsed_wall_time_s": 0.1,
                    "n_simulations": 10,
                    "throughput_sims_per_s": 100.0,
                    "final_quality_wasserstein": 1.0,
                    "final_n_particles": 10,
                    "final_tolerance": 1.0,
                    "state_kind": "accepted_prefix",
                    "test_mode": False,
                }
            ],
            fallback_budget_rows=[],
            fallback_records=[],
        )
        assert captured["stop_policy_by_method"] == {
            "async_propulate_abc": "wall_time_exact",
            "abc_smc_baseline": "wall_time_exact",
        }

    def test_skip_finalize_leaves_only_shards(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "scaling.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 60, "k": 10, "tol_init": 1_000_000_000.0},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"scaling_curve": False, "efficiency": False},
            top_level_updates={
                "scaling": {
                    "worker_counts": [1],
                    "test_worker_counts": [1],
                    "k_values": [10],
                    "test_k_values": [10],
                    "wall_time_budgets_s": [0.05],
                    "wall_time_limit_s": 0.05,
                    "max_simulations_policy": {"min_total": 60, "per_worker": 10, "k_factor": 1},
                }
            },
        )
        config_path = test_helpers.write_config(tmp_path, "scaling_skip_finalize.json", cfg)

        test_helpers.run_runner_main(
            "scaling_runner.py",
            config_path,
            tmp_path,
            extra_args=("--skip-finalize",),
        )

        data_dir = tmp_path / "scaling" / "data"
        assert (data_dir / "throughput_summary_w1_k10.csv").exists()
        assert not (data_dir / "throughput_summary.csv").exists()
        assert not (data_dir / "raw_results.csv").exists()

    def test_scaling_runner_does_not_preclean_combo_artifacts(self, tmp_path, monkeypatch):
        module = test_helpers.import_runner_module("scaling_runner.py")
        cfg = test_helpers.make_fast_runner_config(
            "scaling.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 10, "k": 10},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"scaling_curve": False, "efficiency": False},
            top_level_updates={
                "scaling": {
                    "worker_counts": [1],
                    "test_worker_counts": [1],
                    "k_values": [10],
                    "test_k_values": [10],
                    "wall_time_budgets_s": [0.05],
                    "wall_time_limit_s": 0.05,
                    "max_simulations_policy": {"min_total": 10, "per_worker": 10, "k_factor": 1},
                }
            },
        )

        cleanup_calls = []
        monkeypatch.setattr(module, "configure_logging", lambda: None)
        monkeypatch.setattr(module, "load_config", lambda *args, **kwargs: cfg)
        monkeypatch.setattr(module, "is_root_rank", lambda: True)
        monkeypatch.setattr(
            module,
            "make_benchmark",
            lambda benchmark_cfg: types.SimpleNamespace(
                simulate=lambda params, seed: 0.0,
                limits={"x": (-1.0, 1.0)},
            ),
        )
        monkeypatch.setattr(module, "_cleanup_combo_artifacts", lambda *args, **kwargs: cleanup_calls.append(1))

        module.main(
            [
                "--config",
                str(tmp_path / "scaling_no_preclean.json"),
                "--output-dir",
                str(tmp_path),
                "--test",
                "--skip-finalize",
            ]
        )

        assert cleanup_calls == []

    def test_scaling_runner_test_mode_honors_explicit_n_workers(self, tmp_path, monkeypatch):
        module = test_helpers.import_runner_module("scaling_runner.py")
        cfg = test_helpers.make_fast_runner_config(
            "scaling.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 10, "k": 10},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"scaling_curve": False, "efficiency": False},
            top_level_updates={
                "scaling": {
                    "worker_counts": [1, 4, 48],
                    "test_worker_counts": [1, 4, 48],
                    "k_values": [10],
                    "test_k_values": [10],
                    "wall_time_budgets_s": [0.05],
                    "wall_time_limit_s": 0.05,
                    "max_simulations_policy": {"min_total": 10, "per_worker": 10, "k_factor": 1},
                }
            },
        )

        seen_worker_counts = []
        monkeypatch.setattr(module, "configure_logging", lambda: None)
        monkeypatch.setattr(module, "load_config", lambda *args, **kwargs: cfg)
        monkeypatch.setattr(module, "is_root_rank", lambda: True)
        monkeypatch.setattr(
            module,
            "make_benchmark",
            lambda benchmark_cfg: types.SimpleNamespace(
                simulate=lambda params, seed: 0.0,
                limits={"x": (-1.0, 1.0)},
            ),
        )
        monkeypatch.setattr(
            module,
            "run_method_distributed",
            lambda name, simulate_fn, limits, inference_cfg, output_dir, replicate, seed: (
                seen_worker_counts.append(int(inference_cfg["n_workers"])) or []
            ),
        )

        module.main(
            [
                "--config",
                str(tmp_path / "scaling_filter_workers.json"),
                "--output-dir",
                str(tmp_path),
                "--test",
                "--skip-finalize",
                "--n-workers",
                "48",
            ]
        )

        assert seen_worker_counts
        assert set(seen_worker_counts) == {48}

    def test_scaling_runner_writes_grid_plots(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "scaling.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 60, "k": 10, "tol_init": 1_000_000_000.0},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"scaling_curve": True, "efficiency": True},
            top_level_updates={
                "scaling": {
                    "worker_counts": [1, 4],
                    "test_worker_counts": [1, 4],
                    "k_values": [10, 50],
                    "test_k_values": [10, 50],
                    "wall_time_budgets_s": [0.05, 0.1],
                    "wall_time_limit_s": 0.1,
                    "max_simulations_policy": {"min_total": 60, "per_worker": 10, "k_factor": 1},
                }
            },
        )
        config_path = test_helpers.write_config(tmp_path, "scaling_plots.json", cfg)

        test_helpers.run_runner_main("scaling_runner.py", config_path, tmp_path)

        plots_dir = tmp_path / "scaling" / "plots"
        assert (plots_dir / "throughput_vs_workers__rejection_abc.pdf").exists()
        assert (plots_dir / "efficiency_vs_workers__rejection_abc.pdf").exists()
        assert (plots_dir / "quality_at_budget__rejection_abc__T0p1.pdf").exists()
        assert (plots_dir / "throughput_vs_workers__all_methods_all_k_data.csv").exists()

    # --- Phase 3 tests: fair comparison summaries ---

    def test_time_to_quality_rows_returns_nan_when_threshold_never_reached(self):
        module = test_helpers.import_runner_module("scaling_runner.py")
        budget_rows = [
            {
                "base_method": "async_propulate_abc",
                "method_variant": "async_propulate_abc__k10__w1",
                "k": "10",
                "n_workers": "1",
                "replicate": "0",
                "seed": "1",
                "budget_s": "0.1",
                "attempts_by_budget": "5",
                "quality_wasserstein_by_budget": "2.0",
                "test_mode": "False",
            },
            {
                "base_method": "async_propulate_abc",
                "method_variant": "async_propulate_abc__k10__w1",
                "k": "10",
                "n_workers": "1",
                "replicate": "0",
                "seed": "1",
                "budget_s": "0.2",
                "attempts_by_budget": "10",
                "quality_wasserstein_by_budget": "1.5",
                "test_mode": "False",
            },
        ]
        rows = module._time_to_quality_rows(budget_rows, quality_thresholds=[0.5])

        assert len(rows) == 1
        assert math.isnan(float(rows[0]["time_to_quality_s"])), (
            "time_to_quality_s must be NaN when quality threshold is never reached"
        )
        assert math.isnan(float(rows[0]["realized_attempts_at_threshold"]))

    def test_time_to_quality_rows_returns_first_budget_meeting_threshold(self):
        module = test_helpers.import_runner_module("scaling_runner.py")
        budget_rows = [
            {
                "base_method": "rejection_abc",
                "method_variant": "rejection_abc__k10__w1",
                "k": "10",
                "n_workers": "1",
                "replicate": "0",
                "seed": "1",
                "budget_s": "0.1",
                "attempts_by_budget": "5",
                "quality_wasserstein_by_budget": "2.0",
                "test_mode": "False",
            },
            {
                "base_method": "rejection_abc",
                "method_variant": "rejection_abc__k10__w1",
                "k": "10",
                "n_workers": "1",
                "replicate": "0",
                "seed": "1",
                "budget_s": "0.2",
                "attempts_by_budget": "12",
                "quality_wasserstein_by_budget": "0.8",
                "test_mode": "False",
            },
            {
                "base_method": "rejection_abc",
                "method_variant": "rejection_abc__k10__w1",
                "k": "10",
                "n_workers": "1",
                "replicate": "0",
                "seed": "1",
                "budget_s": "0.3",
                "attempts_by_budget": "18",
                "quality_wasserstein_by_budget": "0.4",
                "test_mode": "False",
            },
        ]
        rows = module._time_to_quality_rows(budget_rows, quality_thresholds=[1.0])

        assert len(rows) == 1
        assert float(rows[0]["time_to_quality_s"]) == 0.2, (
            "should return the first (smallest) budget_s that meets the threshold"
        )
        assert int(rows[0]["realized_attempts_at_threshold"]) == 12

    def test_time_to_quality_fieldnames_are_complete(self):
        module = test_helpers.import_runner_module("scaling_runner.py")
        required = {"time_to_quality_s", "realized_attempts_at_threshold", "quality_threshold"}
        assert required <= set(module._TIME_TO_QUALITY_FIELDNAMES)

    # --- Phase 1 regression tests: freeze run4 measurement failures ---

    def test_worker_utilization_never_exceeds_one(self, monkeypatch):
        """Regression: population_particle timing must not inflate worker_utilization > 1.

        With the bug, sim_intervals includes population_particle records whose
        [sim_start_time, sim_end_time] spans the entire generation interval (0.0–1.0).
        Ten such records give active_time = 10 * 1.0 = 10.0 on top of the 1.0 from
        simulation_attempt records → worker_utilization = 11.0 with 1 worker/1 s elapsed.
        After the fix only simulation_attempt records are counted: utilization = 1.0.
        """
        module = test_helpers.import_runner_module("scaling_runner.py")
        monkeypatch.setattr(module, "final_state_results", lambda records, archive_size: [])

        # 10 simulation_attempt records tiling [0.0, 1.0] exactly (no overlap).
        attempt_records = [
            test_helpers.ParticleRecord(
                method="abc_smc_baseline",
                replicate=0,
                seed=1,
                step=i,
                params={"mu": 0.0},
                loss=0.0,
                weight=1.0,
                tolerance=1.0,
                wall_time=float(i + 1) / 10.0,
                sim_start_time=float(i) / 10.0,
                sim_end_time=float(i + 1) / 10.0,
                generation=0,
                record_kind="simulation_attempt",
                attempt_count=i + 1,
            )
            for i in range(10)
        ]
        # 10 population_particle records sharing the full generation interval [0.0, 1.0].
        # These inflate active_time by 10 * 1.0 = 10.0 when the filter is missing.
        particle_records = [
            test_helpers.ParticleRecord(
                method="abc_smc_baseline",
                replicate=0,
                seed=1,
                step=10 + i,
                params={"mu": float(i) / 10.0},
                loss=float(i) / 10.0,
                weight=0.1,
                tolerance=1.0,
                wall_time=1.0,
                sim_start_time=0.0,
                sim_end_time=1.0,
                generation=0,
                record_kind="population_particle",
                attempt_count=10,
            )
            for i in range(10)
        ]

        row = module._final_summary_row(
            attempt_records + particle_records,
            base_method="abc_smc_baseline",
            method_variant="abc_smc_baseline__k10__w1",
            k=10,
            n_workers=1,
            replicate=0,
            seed=1,
            requested_max_simulations=100,
            max_wall_time_s=None,
            test_mode=True,
            true_params={},
            stop_policy="wall_time_exact",
        )

        assert row["worker_utilization"] <= 1.0, (
            f"worker_utilization={row['worker_utilization']:.3f} > 1.0: "
            "population_particle records are being included in active-time accounting"
        )

    def test_throughput_summary_exposes_realized_attempts(self):
        """Regression: throughput_summary.csv must include a realized_attempts column."""
        module = test_helpers.import_runner_module("scaling_runner.py")
        assert "realized_attempts" in module._THROUGHPUT_FIELDNAMES, (
            "realized_attempts column is missing from _THROUGHPUT_FIELDNAMES; "
            "the CSV cannot support budget-matched comparisons"
        )

    def test_throughput_summary_exposes_posterior_samples(self):
        """Regression: throughput_summary.csv must include a posterior_samples column."""
        module = test_helpers.import_runner_module("scaling_runner.py")
        assert "posterior_samples" in module._THROUGHPUT_FIELDNAMES, (
            "posterior_samples column is missing from _THROUGHPUT_FIELDNAMES; "
            "the CSV cannot support budget-matched comparisons"
        )

    def test_budget_summary_uses_sim_end_time_for_completed_work(self):
        module = test_helpers.import_runner_module("scaling_runner.py")
        records = [
            test_helpers.ParticleRecord(
                method="abc_smc_baseline",
                replicate=0,
                seed=1,
                step=1,
                params={"mu": 0.0},
                loss=0.0,
                weight=1.0,
                tolerance=2.0,
                wall_time=0.1,
                sim_start_time=0.0,
                sim_end_time=0.6,
                generation=0,
                record_kind="simulation_attempt",
                attempt_count=1,
            ),
            test_helpers.ParticleRecord(
                method="abc_smc_baseline",
                replicate=0,
                seed=1,
                step=2,
                params={"mu": 0.1},
                loss=0.1,
                weight=1.0,
                tolerance=1.0,
                wall_time=0.7,
                sim_start_time=0.6,
                sim_end_time=0.7,
                generation=0,
                record_kind="simulation_attempt",
                attempt_count=2,
            ),
        ]

        assert module._attempt_count_upto(records, 0.5) == 0
        assert module._attempt_count_upto(records, 0.65) == 1
        assert math.isnan(module._best_tolerance_upto(records, 0.5))
        assert module._best_tolerance_upto(records, 0.65) == 2.0


class TestTimingComparison:
    def test_small_mode_estimate_is_included_in_comparison(self, tmp_path):
        from async_abc.utils.runner import write_timing_comparison_csv

        exp_dir = tmp_path / "gaussian_mean" / "data"
        exp_dir.mkdir(parents=True)
        with open(exp_dir / "timing.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "experiment_name",
                    "elapsed_s",
                    "estimated_full_s",
                    "estimated_full_unsharded_s",
                    "estimated_full_sharded_wall_s",
                    "aggregate_compute_s",
                    "test_mode",
                    "run_mode",
                    "timestamp",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "experiment_name": "gaussian_mean",
                    "elapsed_s": "10.0",
                    "estimated_full_s": "100.0",
                    "estimated_full_unsharded_s": "100.0",
                    "estimated_full_sharded_wall_s": "",
                    "aggregate_compute_s": "10.0",
                    "test_mode": "False",
                    "run_mode": "small",
                    "timestamp": "2026-03-25T10:00:00",
                }
            )
            writer.writerow(
                {
                    "experiment_name": "gaussian_mean",
                    "elapsed_s": "95.0",
                    "estimated_full_s": "",
                    "estimated_full_unsharded_s": "",
                    "estimated_full_sharded_wall_s": "",
                    "aggregate_compute_s": "95.0",
                    "test_mode": "False",
                    "run_mode": "full",
                    "timestamp": "2026-03-25T11:00:00",
                }
            )

        write_timing_comparison_csv(tmp_path)

        rows = _rows(tmp_path / "timing_comparison.csv")
        assert rows
        assert rows[0]["test_run_mode"] == "small"
        assert rows[0]["estimated_full_s"] == "100.0"
        assert rows[0]["actual_elapsed_s"] == "95.0"


class TestRuntimeHeterogeneityRunner:
    def test_creates_gantt_plot(self, runtime_heterogeneity_runner_artifact):
        plots_dir = runtime_heterogeneity_runner_artifact["root"] / "runtime_heterogeneity" / "plots"
        assert (plots_dir / "worker_gantt.pdf").exists()

    def test_writes_runtime_debug_summary(self, runtime_heterogeneity_runner_artifact):
        data_dir = runtime_heterogeneity_runner_artifact["root"] / "runtime_heterogeneity" / "data"
        rows = _rows(data_dir / "runtime_debug_summary.csv")
        assert rows
        assert {
            "method",
            "base_method",
            "replicate",
            "worker_id",
            "n_attempts",
            "total_busy_s",
            "elapsed_wall_s",
            "active_span_s",
        } <= set(rows[0].keys())

    # ── Phase 1: replicate-specific delay seed ────────────────────────────────

    def test_delay_patterns_differ_by_seed(self):
        """_make_heterogeneous_simulate with different seeds produces different delays."""
        from unittest.mock import patch

        module = test_helpers.import_runner_module("runtime_heterogeneity_runner.py")

        def fake_sim(params, seed=0):
            return 0.0

        delays_a: list[float] = []
        delays_b: list[float] = []

        with patch("time.sleep", side_effect=delays_a.append):
            wrapped_a = module._make_heterogeneous_simulate(
                fake_sim, mu=0.0, sigma=1.0, seed=10, test_mode=False
            )
            for i in range(25):
                wrapped_a({"mu": float(i)}, seed=i * 7 + 1)

        with patch("time.sleep", side_effect=delays_b.append):
            wrapped_b = module._make_heterogeneous_simulate(
                fake_sim, mu=0.0, sigma=1.0, seed=99, test_mode=False
            )
            for i in range(25):
                wrapped_b({"mu": float(i)}, seed=i * 7 + 1)

        assert delays_a != delays_b, "Different seeds must produce different delay sequences"

    def test_delay_seed_is_replicate_specific_in_runner(self, tmp_path):
        """Runner passes per-replicate delay seed (not a fixed constant)."""
        from unittest.mock import patch

        module = test_helpers.import_runner_module("runtime_heterogeneity_runner.py")
        captured_seeds: list[int] = []
        original_make = module._make_heterogeneous_simulate

        def recording_make(simulate_fn, mu, sigma, seed, test_mode=False):
            captured_seeds.append(seed)
            return original_make(simulate_fn, mu, sigma, seed, test_mode=True)

        cfg = test_helpers.make_fast_runner_config(
            "runtime_heterogeneity.json",
            methods=["timed_fake"],
            inference_overrides={"max_simulations": 10, "k": 5},
            execution_overrides={"n_replicates": 2, "base_seed": 0},
            plots={"idle_fraction": False, "throughput_over_time": False,
                   "idle_fraction_comparison": False, "gantt": False},
            top_level_updates={"heterogeneity": {"distribution": "lognormal", "mu": 0.0,
                                                  "sigma_levels": [0.5]}},
        )
        config_path = test_helpers.write_config(tmp_path, "cfg.json", cfg)
        output_dir = tmp_path / "out"

        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            with patch.object(module, "_make_heterogeneous_simulate", recording_make):
                module.main(["--config", str(config_path), "--output-dir", str(output_dir)])

        # With 2 replicates and 1 sigma level → 2 calls with different seeds
        assert len(captured_seeds) == 2, f"Expected 2 delay seeds, got {captured_seeds}"
        assert captured_seeds[0] != captured_seeds[1], (
            f"Replicates must get different delay seeds; got {captured_seeds}"
        )

    # ── Phase 2: base_delay_s calibration ────────────────────────────────────

    def test_base_delay_s_in_config_overrides_mu(self, tmp_path):
        """base_delay_s config key sets median delay (mu = log(base_delay_s))."""
        import math
        from unittest.mock import patch

        module = test_helpers.import_runner_module("runtime_heterogeneity_runner.py")
        captured_seeds: list[int] = []
        original_make = module._make_heterogeneous_simulate

        def recording_make(simulate_fn, mu, sigma, seed, test_mode=False):
            captured_seeds.append(mu)
            return original_make(simulate_fn, mu, sigma, seed, test_mode=True)

        cfg = test_helpers.make_fast_runner_config(
            "runtime_heterogeneity.json",
            methods=["timed_fake"],
            inference_overrides={"max_simulations": 10, "k": 5},
            execution_overrides={"n_replicates": 1, "base_seed": 0},
            plots={"idle_fraction": False, "throughput_over_time": False,
                   "idle_fraction_comparison": False, "gantt": False},
            replace_top_level={"heterogeneity": {"distribution": "lognormal",
                                                  "base_delay_s": 2.5,
                                                  "sigma_levels": [0.5]}},
        )
        config_path = test_helpers.write_config(tmp_path, "cfg.json", cfg)
        output_dir = tmp_path / "out"

        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            with patch.object(module, "_make_heterogeneous_simulate", recording_make):
                module.main(["--config", str(config_path), "--output-dir", str(output_dir)])

        assert captured_seeds, "Expected at least one _make_heterogeneous_simulate call"
        expected_mu = math.log(2.5)
        assert all(abs(mu - expected_mu) < 1e-9 for mu in captured_seeds), (
            f"Expected mu=log(2.5)={expected_mu:.4f} but got {captured_seeds}"
        )

    # ── Phase 4: speedup summary CSV ─────────────────────────────────────────

    def test_writes_speedup_summary_csv(self, runtime_heterogeneity_runner_artifact):
        data_dir = runtime_heterogeneity_runner_artifact["root"] / "runtime_heterogeneity" / "data"
        csv_path = data_dir / "speedup_summary.csv"
        assert csv_path.exists(), "speedup_summary.csv must be written by the runner"
        rows = _rows(csv_path)
        assert rows, "speedup_summary.csv must contain data rows"
        assert {"sigma", "base_method", "median_completion_time_s",
                "speedup_vs_abc_smc_baseline"} <= set(rows[0].keys())

    def test_writes_runtime_performance_summary_csv(self, runtime_heterogeneity_runner_artifact):
        data_dir = runtime_heterogeneity_runner_artifact["root"] / "runtime_heterogeneity" / "data"
        rows = _rows(data_dir / "runtime_performance_summary.csv")
        assert rows
        assert {
            "sigma",
            "base_method",
            "replicate",
            "elapsed_wall_time_s",
            "total_attempts",
            "final_posterior_size",
            "final_quality_wasserstein",
            "throughput_sims_per_s",
            "utilization_loss_fraction",
        } <= set(rows[0].keys())
        util_values = [
            float(row["utilization_loss_fraction"])
            for row in rows
            if row.get("utilization_loss_fraction") not in ("", "nan", "NaN")
        ]
        assert util_values
        assert all(math.isfinite(value) for value in util_values)
        assert all(0.0 <= value <= 1.0 for value in util_values)

    # ── Phase 5: no cluttered combined benchmark diagnostics ──────────────────

    def test_no_cluttered_benchmark_diagnostics_plot(self, runtime_heterogeneity_runner_artifact):
        """Runner must not call plot_benchmark_diagnostics on combined multi-sigma records."""
        plots_dir = runtime_heterogeneity_runner_artifact["root"] / "runtime_heterogeneity" / "plots"
        # If plot_benchmark_diagnostics were called on combined records, it would produce
        # a posterior_comparison.pdf with 10 methods (2 methods × 5 sigma levels).
        # The heterogeneity runner should NOT produce this plot.
        assert not (plots_dir / "posterior_comparison.pdf").exists()


class TestStragglerRunner:
    def test_resolves_effective_straggler_worker_id_by_backend(self):
        module = test_helpers.import_runner_module("straggler_runner.py")

        assert module._resolve_effective_straggler_worker_id(
            "async_propulate_abc",
            0,
            world_size=16,
        ) == "0"
        assert module._resolve_effective_straggler_worker_id(
            "abc_smc_baseline",
            0,
            world_size=16,
        ) == "1"
        assert module._resolve_effective_straggler_worker_id(
            "pyabc_smc",
            2,
            world_size=16,
        ) == "3"
        assert module._resolve_effective_straggler_worker_id(
            "async_propulate_abc",
            0,
            world_size=1,
        ) == "0"

    def test_straggler_runner_fails_when_resolved_worker_is_missing(self, tmp_path, monkeypatch):
        module = test_helpers.import_runner_module("straggler_runner.py")
        cfg = {
            "experiment_name": "straggler",
            "benchmark": {
                "name": "gaussian_mean",
                "observed_data_seed": 42,
                "n_obs": 20,
                "true_mu": 0.0,
                "sigma_obs": 1.0,
                "prior_low": -5.0,
                "prior_high": 5.0,
            },
            "methods": ["timed_fake"],
            "inference": {
                "max_simulations": 10,
                "n_workers": 1,
                "k": 5,
                "tol_init": 5.0,
                "n_generations": 2,
                "scheduler_type": "acceptance_rate",
                "perturbation_scale": 0.8,
            },
            "execution": {
                "n_replicates": 1,
                "base_seed": 1,
            },
            "straggler": {
                "straggler_rank": 0,
                "base_sleep_s": 0.1,
                "slowdown_factor": [1],
            },
            "plots": {
                "throughput_vs_slowdown": False,
                "gantt": False,
            },
        }
        config_path = test_helpers.write_config(tmp_path, "straggler_missing_worker.json", cfg)
        monkeypatch.setattr(module, "_resolve_effective_straggler_worker_id", lambda *args, **kwargs: "9")

        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            with pytest.raises(RuntimeError, match="never observed"):
                module.main(
                    [
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(tmp_path),
                    ]
                )

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

    def test_throughput_csv_exports_active_and_elapsed_wall_time(self, straggler_runner_artifact):
        csv_path = (
            straggler_runner_artifact["root"]
            / "straggler"
            / "data"
            / "throughput_vs_slowdown_summary.csv"
        )
        rows = _rows(csv_path)
        assert rows
        assert "active_wall_time_s" in rows[0]
        assert "elapsed_wall_time_s" in rows[0]
        assert "effective_straggler_worker_id" in rows[0]
        assert "final_quality_wasserstein" in rows[0]
        assert "utilization_loss_fraction" in rows[0]
        util_values = [
            float(row["utilization_loss_fraction"])
            for row in rows
            if row.get("utilization_loss_fraction") not in ("", "nan", "NaN")
        ]
        assert util_values
        assert all(math.isfinite(value) for value in util_values)
        assert all(0.0 <= value <= 1.0 for value in util_values)

    def test_throughput_plot_metadata_is_complete(self, straggler_runner_artifact):
        meta_path = (
            straggler_runner_artifact["root"]
            / "straggler"
            / "plots"
            / "throughput_vs_slowdown_meta.json"
        )
        meta = json.loads(meta_path.read_text())
        assert meta["plot_name"] == "throughput_vs_slowdown"
        assert meta["summary_plot"] is True
        assert meta["experiment_name"] == "straggler"
        assert meta["benchmark"] is False
        assert meta["paper_primary"] is True
        assert meta["summary_source"] == "throughput_vs_slowdown_summary.csv"

    def test_writes_runtime_debug_summary(self, straggler_runner_artifact):
        csv_path = (
            straggler_runner_artifact["root"]
            / "straggler"
            / "data"
            / "runtime_debug_summary.csv"
        )
        rows = _rows(csv_path)
        assert rows
