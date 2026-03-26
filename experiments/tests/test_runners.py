"""Tests for experiment runner scripts."""
import copy
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

    def test_rerun_is_idempotent(self, gaussian_runner_artifact, tmp_path):
        test_helpers.copy_output_tree(gaussian_runner_artifact["root"], tmp_path)
        config_path = test_helpers.clone_artifact_config(gaussian_runner_artifact, tmp_path)

        test_helpers.run_runner_main("gaussian_mean_runner.py", config_path, tmp_path)

        csv_path = tmp_path / "gaussian_mean" / "data" / "raw_results.csv"
        assert csv_path.exists()

    def test_creates_phase3_plots(self, gaussian_runner_artifact):
        plots_dir = gaussian_runner_artifact["root"] / "gaussian_mean" / "plots"
        data_dir = gaussian_runner_artifact["root"] / "gaussian_mean" / "data"
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
    def test_creates_throughput_csv(self, scaling_runner_artifact):
        data_dir = scaling_runner_artifact["root"] / "scaling" / "data"
        assert any(data_dir.glob("throughput*.csv"))

    def test_throughput_csv_has_expected_worker_counts(self, scaling_runner_artifact):
        csv_path = scaling_runner_artifact["root"] / "scaling" / "data" / "throughput_summary.csv"
        rows = _rows(csv_path)
        assert {int(row["n_workers"]) for row in rows} == {1, 4}

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
                "n_workers": 4,
                "method": "async_propulate_abc",
                "replicate": 0,
                "seed": 2,
                "n_simulations": 20,
                "wall_time_s": 2.0,
                "throughput_sims_per_s": 10.0,
                "test_mode": False,
            },
            {
                "n_workers": 1,
                "method": "async_propulate_abc",
                "replicate": 0,
                "seed": 1,
                "n_simulations": 10,
                "wall_time_s": 2.0,
                "throughput_sims_per_s": 5.0,
                "test_mode": False,
            },
        ]
        captured = {}
        monkeypatch.setattr(
            module,
            "write_metadata",
            lambda _output_dir, _cfg, extra=None: captured.setdefault("worker_counts", extra["worker_counts"]),
        )

        module._write_scaling_shards(output_dir, rows)
        aggregate_rows = module.rebuild_scaling_outputs(
            output_dir,
            {"plots": {}, "experiment_name": "scaling"},
        )

        csv_rows = _rows(output_dir.data / "throughput_summary.csv")
        assert [int(row["n_workers"]) for row in csv_rows] == [1, 4]
        assert [int(row["n_workers"]) for row in aggregate_rows] == [1, 4]
        assert captured["worker_counts"] == [1, 4]


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

    def test_writes_runtime_debug_summary(self, straggler_runner_artifact):
        csv_path = (
            straggler_runner_artifact["root"]
            / "straggler"
            / "data"
            / "runtime_debug_summary.csv"
        )
        rows = _rows(csv_path)
        assert rows
