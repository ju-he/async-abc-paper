"""Tests for sharded execution and submission helpers."""
import csv
import json
import os
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

    def test_publish_directory_atomically_moves_into_empty_target(self, tmp_path):
        from async_abc.utils.sharding import publish_directory_atomically

        source_dir = tmp_path / "source"
        (source_dir / "data").mkdir(parents=True)
        (source_dir / "data" / "payload.txt").write_text("fresh")
        target_dir = tmp_path / "target"

        publish_directory_atomically(source_dir, target_dir)

        assert not source_dir.exists()
        assert (target_dir / "data" / "payload.txt").read_text() == "fresh"

    def test_publish_directory_atomically_restores_existing_target_when_replace_fails(self, tmp_path, monkeypatch):
        from async_abc.utils import sharding

        source_dir = tmp_path / "source"
        (source_dir / "data").mkdir(parents=True)
        (source_dir / "data" / "payload.txt").write_text("fresh")
        target_dir = tmp_path / "target"
        (target_dir / "data").mkdir(parents=True)
        (target_dir / "data" / "payload.txt").write_text("stable")

        real_replace = sharding.os.replace

        def flaky_replace(src, dst):
            src_path = Path(src)
            dst_path = Path(dst)
            if src_path == source_dir and dst_path == target_dir:
                raise OSError("injected publish failure")
            return real_replace(src, dst)

        monkeypatch.setattr(sharding.os, "replace", flaky_replace)

        with pytest.raises(OSError, match="injected publish failure"):
            sharding.publish_directory_atomically(source_dir, target_dir)

        assert (target_dir / "data" / "payload.txt").read_text() == "stable"
        assert (source_dir / "data" / "payload.txt").read_text() == "fresh"

    def test_publish_directory_atomically_leaves_no_backup_artifacts_on_success(self, tmp_path):
        from async_abc.utils.sharding import publish_directory_atomically

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "payload.txt").write_text("fresh")
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        (target_dir / "payload.txt").write_text("stable")

        publish_directory_atomically(source_dir, target_dir)

        assert sorted(path.name for path in tmp_path.iterdir()) == ["target"]
        assert (target_dir / "payload.txt").read_text() == "fresh"


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
        root_timing_rows = _rows(tmp_path / "timing_summary_full.csv")
        assert root_timing_rows
        assert root_timing_rows[-1]["experiment_name"] == "gaussian_mean"

        merge_done = tmp_path / "_shards" / "gaussian_mean" / "runs" / "default" / "merge.done.json"
        assert merge_done.exists()

    def test_gaussian_runner_extend_merges_existing_output_with_new_shard_results(self, tmp_path):
        from async_abc.io.paths import OutputDir
        from async_abc.io.records import ParticleRecord, write_records
        from async_abc.utils.metadata import write_metadata
        from async_abc.utils.sharding import ShardLayout, build_plan_payload

        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["timed_fake"],
            inference_overrides={"max_simulations": 60, "k": 10},
            execution_overrides={"n_replicates": 2, "base_seed": 1},
            plots={},
        )
        config_path = test_helpers.write_config(tmp_path, "gaussian_sharded_extend.json", cfg)

        final_output = OutputDir(tmp_path, "gaussian_mean").ensure()
        write_records(
            final_output.data / "raw_results.csv",
            [
                ParticleRecord(
                    method="timed_fake",
                    replicate=0,
                    seed=1,
                    step=1,
                    params={"mu": 0.1},
                    loss=0.4,
                    weight=0.5,
                    tolerance=1.0,
                    wall_time=0.4,
                    worker_id="0",
                    sim_start_time=0.0,
                    sim_end_time=0.2,
                    generation=0,
                ),
                ParticleRecord(
                    method="timed_fake",
                    replicate=0,
                    seed=1,
                    step=2,
                    params={"mu": -0.1},
                    loss=0.2,
                    weight=0.5,
                    tolerance=0.5,
                    wall_time=0.4,
                    worker_id="1",
                    sim_start_time=0.2,
                    sim_end_time=0.4,
                    generation=1,
                ),
            ],
        )
        write_metadata(final_output, cfg)

        layout = ShardLayout(tmp_path, "gaussian_mean", "extend-run", 0)
        layout.plan_path.parent.mkdir(parents=True, exist_ok=True)
        layout.plan_path.write_text(
            json.dumps(
                build_plan_payload(
                    experiment_name="gaussian_mean",
                    config_path=str(config_path.resolve()),
                    unit_kind="replicate",
                    full_total_units=2,
                    actual_total_units=2,
                    target_total_units=1,
                    requested_num_shards=1,
                    actual_num_shards=1,
                    test_mode=False,
                    small_mode=False,
                    run_mode="full",
                    extend=True,
                    run_id="extend-run",
                    completed_unit_indices=[0],
                    pending_unit_indices=[1],
                    shard_assignments=[[1]],
                    runner_script="gaussian_mean_runner.py",
                )
            )
        )

        with test_helpers.patched_method_registry({"timed_fake": test_helpers.timed_fake_method}):
            test_helpers.run_runner_main(
                "gaussian_mean_runner.py",
                config_path,
                tmp_path,
                extra_args=(
                    "--shard-index",
                    "0",
                    "--num-shards",
                    "1",
                    "--shard-run-id",
                    "extend-run",
                    "--extend",
                ),
            )

        rows = _rows(tmp_path / "gaussian_mean" / "data" / "raw_results.csv")
        key_tuples = [(row["method"], row["replicate"]) for row in rows]

        assert key_tuples.count(("timed_fake", "0")) == 2
        assert key_tuples.count(("timed_fake", "1")) == 2
        assert len(rows) == 4

        metadata = json.loads((tmp_path / "gaussian_mean" / "data" / "metadata.json").read_text())
        assert metadata["completed_replicates"] == [0, 1]
        assert metadata["last_shard_run_id"] == "extend-run"


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
        assert (data_dir / "sbc_trials.jsonl").exists()
        assert (data_dir / "sbc_ranks.csv").exists()
        assert (data_dir / "coverage.csv").exists()
        assert (data_dir / "timing.csv").exists()


class TestShardedSbcFinalizer:
    def test_extend_merges_existing_and_new_trial_records(self, tmp_path, sbc_config_file):
        from async_abc.io.config import load_config
        from async_abc.utils.shard_finalizers import finalize_sbc_experiment
        from async_abc.utils.sharding import ShardLayout

        cfg = load_config(sbc_config_file)
        layout = ShardLayout(tmp_path, "sbc", "extend-run", 0)
        layout.final_output_dir.ensure()
        (layout.final_output_dir.data / "sbc_trials.jsonl").write_text(
            json.dumps(
                {
                    "trial": 0,
                    "method": "rejection_abc",
                    "param": "mu",
                    "true_value": 0.0,
                    "posterior_samples": [0.1, 0.2],
                }
            )
            + "\n"
        )

        plan_path = layout.plan_path
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps({"extend": True}))

        shard_layout = ShardLayout(tmp_path, "sbc", "extend-run", 0)
        shard_layout.shard_output_dir.ensure()
        (shard_layout.shard_output_dir.data / "sbc_trials.jsonl").write_text(
            json.dumps(
                {
                    "trial": 1,
                    "method": "rejection_abc",
                    "param": "mu",
                    "true_value": 1.0,
                    "posterior_samples": [0.3, 0.4],
                }
            )
            + "\n"
        )

        finalize_sbc_experiment(cfg, layout, [shard_layout.shard_output_dir], [{"state": "completed"}])

        merged_path = tmp_path / "sbc" / "data" / "sbc_trials.jsonl"
        merged_lines = [json.loads(line) for line in merged_path.read_text().splitlines() if line.strip()]
        assert len(merged_lines) == 2
        assert {row["trial"] for row in merged_lines} == {0, 1}

    def test_extend_fails_when_existing_finalized_output_lacks_trial_corpus(self, tmp_path, sbc_config_file):
        from async_abc.io.config import load_config
        from async_abc.utils.shard_finalizers import finalize_sbc_experiment
        from async_abc.utils.sharding import ShardLayout

        cfg = load_config(sbc_config_file)
        layout = ShardLayout(tmp_path, "sbc", "legacy-run", 0)
        layout.final_output_dir.ensure()
        layout.plan_path.parent.mkdir(parents=True, exist_ok=True)
        layout.plan_path.write_text(json.dumps({"extend": True}))

        shard_layout = ShardLayout(tmp_path, "sbc", "legacy-run", 0)
        shard_layout.shard_output_dir.ensure()
        (shard_layout.shard_output_dir.data / "sbc_trials.jsonl").write_text(
            json.dumps(
                {
                    "trial": 1,
                    "method": "rejection_abc",
                    "param": "mu",
                    "true_value": 1.0,
                    "posterior_samples": [0.3, 0.4],
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="missing sbc_trials.jsonl"):
            finalize_sbc_experiment(cfg, layout, [shard_layout.shard_output_dir], [{"state": "completed"}])


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

    def test_finalize_benchmark_experiment_preserves_existing_output_when_publish_fails(self, tmp_path, monkeypatch):
        from async_abc.io.paths import OutputDir
        from async_abc.io.records import ParticleRecord, write_records
        from async_abc.utils import sharding
        from async_abc.utils.shard_finalizers import finalize_benchmark_experiment
        from async_abc.utils.sharding import ShardLayout

        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 20, "k": 5},
            execution_overrides={"n_replicates": 1, "base_seed": 1},
            plots={},
        )
        layout = ShardLayout(tmp_path, "gaussian_mean", "default", 0)
        layout.final_output_dir.ensure()
        write_records(
            layout.final_output_dir.data / "raw_results.csv",
            [
                ParticleRecord(
                    method="rejection_abc",
                    replicate=0,
                    seed=1,
                    step=1,
                    params={"mu": 0.0},
                    loss=0.1,
                )
            ],
        )

        shard_dir = OutputDir(layout.shard_root, "gaussian_mean").ensure()
        write_records(
            shard_dir.data / "raw_results.csv",
            [
                ParticleRecord(
                    method="rejection_abc",
                    replicate=0,
                    seed=1,
                    step=1,
                    params={"mu": 1.0},
                    loss=0.2,
                )
            ],
        )

        temp_output_root = layout.run_root / "_merge_tmp" / "gaussian_mean"
        real_replace = sharding.os.replace

        def flaky_replace(src, dst):
            src_path = Path(src)
            dst_path = Path(dst)
            if src_path == temp_output_root and dst_path == layout.final_output_dir.root:
                raise OSError("injected publish failure")
            return real_replace(src, dst)

        monkeypatch.setattr(sharding.os, "replace", flaky_replace)

        with pytest.raises(OSError, match="injected publish failure"):
            finalize_benchmark_experiment(cfg, layout, [shard_dir], [{"state": "completed"}])

        rows = _rows(layout.final_output_dir.data / "raw_results.csv")
        assert len(rows) == 1
        assert float(rows[0]["param_mu"]) == pytest.approx(0.0)

    def test_finalize_only_can_retry_merge_after_finalize_stage_failure(self, tmp_path):
        from async_abc.utils.sharding import ShardLayout, all_shards_completed

        layout = ShardLayout(tmp_path, "gaussian_mean", "default", 0)
        shard0 = ShardLayout(tmp_path, "gaussian_mean", "default", 0)
        shard1 = ShardLayout(tmp_path, "gaussian_mean", "default", 1)
        shard0.shard_output_dir.ensure()
        shard1.shard_output_dir.ensure()

        (shard0.shard_output_dir.data / "raw_results.csv").write_text("method,replicate\nm,0\n")
        (shard1.shard_output_dir.data / "raw_results.csv").write_text("method,replicate\nm,1\n")
        shard0.shard_status_path.parent.mkdir(parents=True, exist_ok=True)
        shard1.shard_status_path.parent.mkdir(parents=True, exist_ok=True)
        shard0.shard_status_path.write_text(json.dumps({"state": "completed"}))
        shard1.shard_status_path.write_text(
            json.dumps(
                {
                    "state": "failed",
                    "traceback": "Traceback...\nmaybe_finalize_sharded_run\nfinalize_experiment_by_name\n",
                }
            )
        )

        assert all_shards_completed(layout, 2) is True


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

    def test_submit_replicate_shards_caps_auto_derived_time_limit(self, tmp_path, monkeypatch):
        submitter = test_helpers.import_runner_module("../jobs/submit_replicate_shards.py")
        monkeypatch.setattr(
            submitter.run_all,
            "EXPERIMENT_REGISTRY",
            {"gaussian_mean": ("gaussian_mean_runner.py", "gaussian_mean.json")},
        )

        data_dir = tmp_path / "gaussian_mean" / "data"
        data_dir.mkdir(parents=True)
        with open(data_dir / "timing.csv", "w", newline="") as f:
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
                    "timestamp",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "experiment_name": "gaussian_mean",
                    "elapsed_s": "1.0",
                    "estimated_full_s": "1.0",
                    "estimated_full_unsharded_s": "1.0",
                    "estimated_full_sharded_wall_s": "9999999.0",
                    "aggregate_compute_s": "1.0",
                    "test_mode": "True",
                    "timestamp": "2026-01-01T00:00:00",
                }
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

        scripts = list((tmp_path / "_jobs" / "gaussian_mean").glob("*/*.sbatch"))
        assert len(scripts) == 1
        script_text = scripts[0].read_text()
        assert "#SBATCH --time=24:00:00" in script_text

    def test_submit_replicate_shards_surfaces_sbatch_stderr(self, tmp_path, monkeypatch):
        submitter = test_helpers.import_runner_module("../jobs/submit_replicate_shards.py")
        monkeypatch.setattr(
            submitter.run_all,
            "EXPERIMENT_REGISTRY",
            {"gaussian_mean": ("gaussian_mean_runner.py", "gaussian_mean.json")},
        )

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=args[0],
                stderr="Requested time limit is invalid for partition batch",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
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
            ],
        )

        with pytest.raises(SystemExit, match="Requested time limit is invalid for partition batch"):
            submitter.main()

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

    def test_submit_replicate_shards_small_uses_small_tier_replicate_count(self, tmp_path, monkeypatch):
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
                "--small",
                "--dry-run",
            ],
        )
        submitter.main()

        plan = json.loads(_plan_paths(tmp_path, "gaussian_mean")[0].read_text())
        assert plan["target_total_units"] == 2
        assert plan["requested_num_shards"] == 2
        assert plan["actual_num_shards"] == 2
        assert plan["small_mode"] is True
        assert plan["run_mode"] == "small"


class TestShardSmokeScript:
    def test_local_sharded_smoke_script(self, tmp_path):
        script = Path(__file__).resolve().parents[1] / "jobs" / "test_sharded.sh"
        env = dict(os.environ)
        env["PYTHON_BIN"] = sys.executable
        result = subprocess.run(
            ["bash", str(script), str(tmp_path / "script_run")],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
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
