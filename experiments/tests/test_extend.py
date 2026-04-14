"""Tests for --extend flag and completed-combination helpers."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers


def _count_csv_rows(csv_path):
    with open(csv_path) as f:
        return sum(1 for _ in csv.DictReader(f))


def _read_key_tuples(csv_path, key_cols):
    with open(csv_path) as f:
        return [tuple(row[c] for c in key_cols) for row in csv.DictReader(f)]


def _extend_runner_config():
    return test_helpers.make_fast_runner_config(
        "gaussian_mean.json",
        methods=["rejection_abc", "timed_fake"],
        inference_overrides={"max_simulations": 80, "k": 10},
        execution_overrides={"n_replicates": 2, "base_seed": 1},
        plots={},
    )


class TestFindCompletedCombinations:
    def test_missing_file_returns_empty_set(self, tmp_path):
        from async_abc.utils.runner import find_completed_combinations

        result = find_completed_combinations(tmp_path / "nonexistent.csv", ["method", "replicate"])
        assert result == set()

    def test_empty_file_returns_empty_set(self, tmp_path):
        from async_abc.utils.runner import find_completed_combinations

        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = find_completed_combinations(csv_path, ["method", "replicate"])
        assert result == set()

    def test_parses_key_tuples_correctly(self, tmp_path):
        from async_abc.utils.runner import find_completed_combinations

        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "method,replicate,seed,loss\n"
            "async_propulate_abc,0,42,1.5\n"
            "async_propulate_abc,0,42,1.3\n"
            "rejection_abc,1,99,2.0\n"
        )
        result = find_completed_combinations(csv_path, ["method", "replicate"])
        assert result == {("async_propulate_abc", "0"), ("rejection_abc", "1")}

    def test_three_key_columns(self, tmp_path):
        from async_abc.utils.runner import find_completed_combinations

        csv_path = tmp_path / "throughput.csv"
        csv_path.write_text(
            "n_workers,method,replicate,throughput_sims_per_s\n"
            "1,async_propulate_abc,0,123.4\n"
            "4,async_propulate_abc,0,456.7\n"
        )
        result = find_completed_combinations(csv_path, ["n_workers", "method", "replicate"])
        assert result == {("1", "async_propulate_abc", "0"), ("4", "async_propulate_abc", "0")}


@pytest.fixture
def extend_runner_config_file(tmp_path):
    cfg = _extend_runner_config()
    return test_helpers.write_config(tmp_path, "gaussian_extend.json", cfg)


class TestExtendBasicRunner:
    def test_extend_on_fresh_dir_runs_normally(self, tmp_path, extend_runner_config_file):
        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            test_helpers.run_runner_main(
                "gaussian_mean_runner.py",
                extend_runner_config_file,
                tmp_path,
                extra_args=("--extend",),
            )

        csv_path = tmp_path / "gaussian_mean" / "data" / "raw_results.csv"
        tuples = set(_read_key_tuples(csv_path, ["method", "replicate"]))
        assert csv_path.exists()
        assert tuples == {
            ("rejection_abc", "0"),
            ("rejection_abc", "1"),
            ("timed_fake", "0"),
            ("timed_fake", "1"),
        }

    def test_extend_skips_completed_combinations(self, tmp_path, extend_runner_config_file):
        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            test_helpers.run_runner_main("gaussian_mean_runner.py", extend_runner_config_file, tmp_path)
            csv_path = tmp_path / "gaussian_mean" / "data" / "raw_results.csv"
            row_count_before = _count_csv_rows(csv_path)

            test_helpers.run_runner_main(
                "gaussian_mean_runner.py",
                extend_runner_config_file,
                tmp_path,
                extra_args=("--extend",),
            )

        assert _count_csv_rows(csv_path) == row_count_before

    def test_extend_no_duplicate_key_tuples(self, tmp_path, extend_runner_config_file):
        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            test_helpers.run_runner_main("gaussian_mean_runner.py", extend_runner_config_file, tmp_path)
            test_helpers.run_runner_main(
                "gaussian_mean_runner.py",
                extend_runner_config_file,
                tmp_path,
                extra_args=("--extend",),
            )

        csv_path = tmp_path / "gaussian_mean" / "data" / "raw_results.csv"
        tuples = set(_read_key_tuples(csv_path, ["method", "replicate"]))
        assert len(tuples) == 4

    def test_extend_runs_only_missing_combinations(self, tmp_path, extend_runner_config_file):
        data_dir = tmp_path / "gaussian_mean" / "data"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "raw_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "replicate", "seed", "step", "loss", "wall_time"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": "rejection_abc",
                    "replicate": "0",
                    "seed": "1",
                    "step": "0",
                    "loss": "1.0",
                    "wall_time": "0.1",
                }
            )

        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            test_helpers.run_runner_main(
                "gaussian_mean_runner.py",
                extend_runner_config_file,
                tmp_path,
                extra_args=("--extend",),
            )

        rows = _read_key_tuples(csv_path, ["method", "replicate"])
        assert rows.count(("rejection_abc", "0")) == 1
        assert set(rows) == {
            ("rejection_abc", "0"),
            ("rejection_abc", "1"),
            ("timed_fake", "0"),
            ("timed_fake", "1"),
        }

    def test_extend_matches_fresh_run_same_seed(self, tmp_path, extend_runner_config_file):
        def _rows_as_set(path, cols):
            with open(path) as f:
                return {tuple(row[c] for c in cols) for row in csv.DictReader(f)}

        key_cols = ["method", "replicate", "seed", "step", "loss"]

        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            # Step a: fresh run — ground truth
            fresh_dir = tmp_path / "fresh"
            fresh_dir.mkdir()
            test_helpers.run_runner_main(
                "gaussian_mean_runner.py", extend_runner_config_file, fresh_dir
            )
            fresh_csv = fresh_dir / "gaussian_mean" / "data" / "raw_results.csv"
            assert fresh_csv.exists()

            # Step b: write partial CSV into extend_dir (only rejection_abc, replicate=0)
            extend_dir = tmp_path / "extended"
            (extend_dir / "gaussian_mean" / "data").mkdir(parents=True)
            extend_csv = extend_dir / "gaussian_mean" / "data" / "raw_results.csv"

            with open(fresh_csv) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                partial_rows = [
                    row for row in reader
                    if row["method"] == "rejection_abc" and row["replicate"] == "0"
                ]

            with open(extend_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(partial_rows)

            # Step c: run --extend to complete the remaining combinations
            test_helpers.run_runner_main(
                "gaussian_mean_runner.py",
                extend_runner_config_file,
                extend_dir,
                extra_args=("--extend",),
            )

        # Assertions
        assert set(_read_key_tuples(extend_csv, ["method", "replicate"])) == {
            ("rejection_abc", "0"),
            ("rejection_abc", "1"),
            ("timed_fake", "0"),
            ("timed_fake", "1"),
        }

        fresh_set = _rows_as_set(fresh_csv, key_cols)
        extend_set = _rows_as_set(extend_csv, key_cols)
        assert extend_set == fresh_set

        assert _count_csv_rows(extend_csv) == _count_csv_rows(fresh_csv)


class TestExtendScalingRunner:
    def test_extend_does_not_duplicate_rows(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "scaling.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 60, "k": 10, "tol_init": 1_000_000_000.0},
            execution_overrides={"n_replicates": 2, "base_seed": 1},
            plots={"scaling_curve": False, "efficiency": False},
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
        config_path = test_helpers.write_config(tmp_path, "scaling_extend.json", cfg)

        test_helpers.run_runner_main("scaling_runner.py", config_path, tmp_path)
        csv_path = tmp_path / "scaling" / "data" / "throughput_summary.csv"
        count_before = _count_csv_rows(csv_path)

        test_helpers.run_runner_main(
            "scaling_runner.py",
            config_path,
            tmp_path,
            extra_args=("--extend",),
        )

        assert _count_csv_rows(csv_path) == count_before
        tuples = _read_key_tuples(csv_path, ["n_workers", "k", "base_method", "replicate"])
        assert len(tuples) == len(set(tuples))


class TestExtendSensitivityRunner:
    def test_extend_does_not_add_rows_to_existing_variant_csvs(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "sensitivity.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 60, "k": 10},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={"sensitivity_heatmap": False},
            replace_top_level={
                "sensitivity_grid": {
                    "tol_init_multiplier": [1.0, 2.0],
                }
            },
        )
        config_path = test_helpers.write_config(tmp_path, "sensitivity_extend.json", cfg)

        test_helpers.run_runner_main("sensitivity_runner.py", config_path, tmp_path)
        data_dir = tmp_path / "sensitivity" / "data"
        counts_before = {path: _count_csv_rows(path) for path in data_dir.glob("sensitivity_*.csv")}

        test_helpers.run_runner_main(
            "sensitivity_runner.py",
            config_path,
            tmp_path,
            extra_args=("--extend",),
        )

        for path, count_before in counts_before.items():
            assert _count_csv_rows(path) == count_before


class TestExtendOrchestrator:
    def test_orchestrator_accepts_extend_flag(self, tmp_path, monkeypatch):
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 60, "k": 10},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={},
        )
        config_path = test_helpers.write_config(tmp_path, "gaussian_run_all.json", cfg)
        run_all = test_helpers.import_runner_module("../run_all_paper_experiments.py")
        monkeypatch.setattr(run_all, "CONFIGS_DIR", tmp_path)
        monkeypatch.setattr(
            run_all,
            "EXPERIMENT_REGISTRY",
            {"gaussian_mean": ("gaussian_mean_runner.py", config_path.name)},
        )

        run_all.main(
            [
                "--output-dir",
                str(tmp_path / "results"),
                "--test",
                "--extend",
                "--experiments",
                "gaussian_mean",
            ]
        )

        assert (tmp_path / "results" / "timing_summary_test.csv").exists()
