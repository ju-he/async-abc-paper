"""Tests for sharded execution and submission helpers."""
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers


def _rows(csv_path: Path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


class TestShardHelpers:
    def test_split_indices_balanced_and_contiguous(self):
        from async_abc.utils.sharding import split_indices

        assert split_indices(0, 3) == [[], [], []]
        assert split_indices(5, 2) == [[0, 1, 2], [3, 4]]
        assert split_indices(5, 4) == [[0, 1], [2], [3], [4]]

    def test_estimate_sharded_wall_time_uses_largest_shard(self):
        from async_abc.utils.sharding import estimate_sharded_wall_time

        assert estimate_sharded_wall_time(100.0, total_units=10, requested_num_shards=4) == 30.0


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

        merge_done = tmp_path / "_shards" / "gaussian_mean" / "merge.done.json"
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

        plan = json.loads((tmp_path / "_shards" / "gaussian_mean" / "plan.json").read_text())
        assert plan["requested_num_shards"] == 4
        assert plan["actual_num_shards"] == 1

        scripts = list((tmp_path / "_jobs" / "gaussian_mean").glob("*.sbatch"))
        assert len(scripts) == 1


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
