"""Tests for sharded execution and submission helpers."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers


def _rows(csv_path: Path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _plan_paths(root: Path, experiment_name: str):
    return sorted((root / "_shards" / experiment_name / "runs").glob("*/plan.json"))


class TestShardHelpers:
    def test_split_indices_balanced_and_contiguous(self):
        from async_abc.utils.sharding import split_indices

        assert split_indices(0, 3) == [[], [], []]
        assert split_indices(5, 2) == [[0, 1, 2], [3, 4]]
        assert split_indices(5, 4) == [[0, 1], [2], [3], [4]]

    def test_estimate_sharded_wall_time_uses_largest_shard(self):
        from async_abc.utils.sharding import estimate_sharded_wall_time

        assert estimate_sharded_wall_time(100.0, total_units=10, requested_num_shards=4) == 30.0

    def test_detect_completed_replicates_with_holes(self, tmp_path):
        from async_abc.utils.metadata import write_metadata
        from async_abc.utils.sharding import detect_completed_replicates
        from async_abc.io.paths import OutputDir

        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 20, "k": 5},
            execution_overrides={"n_replicates": 5, "base_seed": 1},
            plots={},
        )
        output_dir = OutputDir(tmp_path, "gaussian_mean").ensure()
        with open(output_dir.data / "raw_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "replicate", "step", "wall_time"],
            )
            writer.writeheader()
            for replicate in (0, 1, 3, 4):
                writer.writerow({"method": "rejection_abc", "replicate": replicate, "step": 1, "wall_time": 0.1})
        write_metadata(output_dir, cfg)

        assert detect_completed_replicates(tmp_path, cfg) == [0, 1, 3, 4]


class TestShardedBenchmarkRunner:
    def test_gaussian_runner_merges_two_shards(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 60, "k": 10},
            execution_overrides={"n_replicates": 3, "base_seed": 1},
            plots={},
        )
        config_path = test_helpers.write_config(tmp_path, "gaussian_sharded.json", cfg)

        test_helpers.run_runner_main(
            "gaussian_mean_runner.py",
            config_path,
            tmp_path,
            extra_args=("--shard-index", "0", "--num-shards", "2"),
        )
        test_helpers.run_runner_main(
            "gaussian_mean_runner.py",
            config_path,
            tmp_path,
            extra_args=("--shard-index", "1", "--num-shards", "2"),
        )

        final_csv = tmp_path / "gaussian_mean" / "data" / "raw_results.csv"
        assert final_csv.exists()
        replicates = {int(row["replicate"]) for row in _rows(final_csv)}
        assert replicates == {0, 1, 2}

        timing_rows = _rows(tmp_path / "gaussian_mean" / "data" / "timing.csv")
        assert timing_rows
        assert "estimated_full_unsharded_s" in timing_rows[-1]
        assert "aggregate_compute_s" in timing_rows[-1]
        root_timing_rows = _rows(tmp_path / "timing_summary.csv")
        assert root_timing_rows
        assert root_timing_rows[-1]["experiment_name"] == "gaussian_mean"

        merge_done = tmp_path / "_shards" / "gaussian_mean" / "runs" / "default" / "merge.done.json"
        assert merge_done.exists()


class TestShardedSbcRunner:
    def test_sbc_runner_merges_trial_shards(self, tmp_path, sbc_config_file):
        test_helpers.run_runner_main(
            "sbc_runner.py",
            sbc_config_file,
            tmp_path,
            extra_args=("--shard-index", "0", "--num-shards", "2"),
        )
        test_helpers.run_runner_main(
            "sbc_runner.py",
            sbc_config_file,
            tmp_path,
            extra_args=("--shard-index", "1", "--num-shards", "2"),
        )

        data_dir = tmp_path / "sbc" / "data"
        assert (data_dir / "sbc_ranks.csv").exists()
        assert (data_dir / "coverage.csv").exists()
        assert (data_dir / "timing.csv").exists()


class TestShardedStragglerRunner:
    def test_straggler_runner_finalizes_single_shard(self, tmp_path, straggler_config_file):
        test_helpers.run_runner_main(
            "straggler_runner.py",
            straggler_config_file,
            tmp_path,
            extra_args=("--shard-index", "0", "--num-shards", "1"),
        )

        data_dir = tmp_path / "straggler" / "data"
        assert (data_dir / "raw_results.csv").exists()
        assert (data_dir / "throughput_vs_slowdown_summary.csv").exists()
        assert (data_dir / "timing.csv").exists()
        assert (tmp_path / "_shards" / "straggler" / "runs" / "default" / "merge.done.json").exists()


class TestShardFailureStatus:
    def test_runner_exception_marks_shard_failed(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["failing_method"],
            inference_overrides={"max_simulations": 10, "k": 5, "n_workers": 1},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={},
        )
        config_path = test_helpers.write_config(tmp_path, "gaussian_failure.json", cfg)

        def failing_method(simulate_fn, limits, inference_cfg, output_dir, replicate, seed):
            raise RuntimeError("boom from fake shard method")

        with test_helpers.patched_method_registry({"failing_method": failing_method}):
            with pytest.raises(RuntimeError, match="boom from fake shard method"):
                test_helpers.run_runner_main(
                    "gaussian_mean_runner.py",
                    config_path,
                    tmp_path,
                    extra_args=("--shard-index", "0", "--num-shards", "1"),
                )

        status_path = tmp_path / "_shards" / "gaussian_mean" / "runs" / "default" / "shard-000" / "status.json"
        status = json.loads(status_path.read_text())

        assert status["state"] == "failed"
        assert status["unit_indices"] == [0]
        assert status["error_type"] == "RuntimeError"
        assert "boom from fake shard method" in status["error_message"]
        assert "Traceback" in status["traceback"]
        assert "started_at_s" in status
        assert "finished_at_s" in status


class TestShardedTestModeMetadata:
    def test_sensitivity_test_mode_metadata_uses_completed_shards(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "sensitivity.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 20, "k": 5, "n_workers": 1},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={},
        )
        config_path = test_helpers.write_config(tmp_path, "sensitivity_test.json", cfg)

        test_helpers.run_runner_main(
            "sensitivity_runner.py",
            config_path,
            tmp_path,
            test_mode=True,
            extra_args=("--shard-index", "0", "--num-shards", "1"),
        )

        metadata = json.loads((tmp_path / "sensitivity" / "data" / "metadata.json").read_text())

        assert metadata["completed_replicates"] == [0]
        assert metadata["completed_replicate_count"] == 1


class TestShardSubmitter:
    def test_submit_replicate_shards_dry_run_writes_plan_and_single_test_shard(self, tmp_path, monkeypatch):
        submitter = test_helpers.import_runner_module("../jobs/submit_replicate_shards.py")
        monkeypatch.setattr(
            submitter.run_all,
            "EXPERIMENT_REGISTRY",
            {"gaussian_mean": ("gaussian_mean_runner.py", "gaussian_mean.json")},
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_replicate_shards.py",
                str(tmp_path),
                "--experiments",
                "gaussian_mean",
                "--jobs-per-experiment",
                "4",
                "--test",
                "--dry-run",
            ],
        )
        submitter.main()

        plan_path = _plan_paths(tmp_path, "gaussian_mean")[0]
        plan = json.loads(plan_path.read_text())
        assert plan["requested_num_shards"] == 4
        assert plan["actual_num_shards"] == 1

        scripts = list((tmp_path / "_jobs" / "gaussian_mean").glob("*/*.sbatch"))
        assert len(scripts) == 1

    def test_submit_replicate_shards_accepts_all_token(self, tmp_path, monkeypatch):
        submitter = test_helpers.import_runner_module("../jobs/submit_replicate_shards.py")
        monkeypatch.setattr(
            submitter.run_all,
            "EXPERIMENT_REGISTRY",
            {
                "gaussian_mean": ("gaussian_mean_runner.py", "gaussian_mean.json"),
                "gandk": ("gandk_runner.py", "gandk.json"),
                "scaling": ("scaling_runner.py", "scaling.json"),
            },
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_replicate_shards.py",
                str(tmp_path),
                "--experiments",
                "all",
                "--jobs-per-experiment",
                "1",
                "--dry-run",
            ],
        )
        submitter.main()

        assert _plan_paths(tmp_path, "gaussian_mean")
        assert _plan_paths(tmp_path, "gandk")
        assert not (tmp_path / "_shards" / "scaling").exists()

    def test_submit_replicate_shards_add_replicates_submits_missing_only(self, tmp_path, monkeypatch):
        from async_abc.io.paths import OutputDir
        from async_abc.utils.metadata import write_metadata

        submitter = test_helpers.import_runner_module("../jobs/submit_replicate_shards.py")
        configs_dir = tmp_path / "configs"
        scripts_dir = tmp_path / "scripts"
        configs_dir.mkdir()
        scripts_dir.mkdir()
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 20, "k": 5, "n_workers": 1},
            execution_overrides={"n_replicates": 5, "base_seed": 1},
            plots={},
        )
        (configs_dir / "gaussian_mean.json").write_text(json.dumps(cfg))
        (scripts_dir / "gaussian_mean_runner.py").write_text("# placeholder\n")
        monkeypatch.setattr(
            submitter.run_all,
            "EXPERIMENT_REGISTRY",
            {"gaussian_mean": ("gaussian_mean_runner.py", "gaussian_mean.json")},
        )
        monkeypatch.setattr(submitter, "EXPERIMENTS_DIR", tmp_path)
        monkeypatch.setattr(submitter, "SCRIPT_DIR", tmp_path / "jobs")

        output_dir = OutputDir(tmp_path, "gaussian_mean").ensure()
        with open(output_dir.data / "raw_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "replicate", "step", "wall_time"])
            writer.writeheader()
            for replicate in (0, 1, 3, 4):
                writer.writerow({"method": "rejection_abc", "replicate": replicate, "step": 1, "wall_time": 0.1})
        write_metadata(output_dir, cfg)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_replicate_shards.py",
                str(tmp_path),
                "--experiments",
                "gaussian_mean",
                "--jobs-per-experiment",
                "3",
                "--add-replicates",
                "--dry-run",
            ],
        )
        submitter.main()

        plan = json.loads(_plan_paths(tmp_path, "gaussian_mean")[0].read_text())
        assert plan["completed_unit_indices"] == [0, 1, 3, 4]
        assert plan["pending_unit_indices"] == [2, 5, 6, 7, 8]
        assert plan["target_total_units"] == 9

    def test_submit_replicate_shards_add_replicates_rejects_test(self, tmp_path, monkeypatch):
        submitter = test_helpers.import_runner_module("../jobs/submit_replicate_shards.py")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_replicate_shards.py",
                str(tmp_path),
                "--experiments",
                "gaussian_mean",
                "--add-replicates",
                "--test",
            ],
        )
        try:
            submitter.main()
        except SystemExit as exc:
            assert "--add-replicates is not supported with --test" in str(exc)
        else:
            raise AssertionError("expected SystemExit")


class TestShardSmokeScript:
    def test_local_sharded_smoke_script(self, tmp_path):
        script = Path(__file__).resolve().parents[1] / "jobs" / "test_sharded.sh"
        result = subprocess.run(
            ["bash", str(script), str(tmp_path / "script_run")],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "Sharded smoke test completed successfully." in result.stdout

    def test_slurm_sharded_smoke_script_has_valid_bash_syntax(self):
        script = Path(__file__).resolve().parents[1] / "jobs" / "test_sharded_slurm.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
