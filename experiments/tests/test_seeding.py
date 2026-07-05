"""Tests for async_abc.utils.seeding."""
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers

from async_abc.utils.seeding import make_seeds, seed_everything


class TestMakeSeeds:
    def test_returns_correct_length(self):
        seeds = make_seeds(5, base=0)
        assert len(seeds) == 5

    def test_returns_list_of_ints(self):
        seeds = make_seeds(3, base=42)
        assert all(isinstance(s, int) for s in seeds)

    def test_deterministic(self):
        seeds_a = make_seeds(4, base=7)
        seeds_b = make_seeds(4, base=7)
        assert seeds_a == seeds_b

    def test_different_base_different_seeds(self):
        seeds_a = make_seeds(4, base=0)
        seeds_b = make_seeds(4, base=1)
        assert seeds_a != seeds_b

    def test_seeds_are_unique(self):
        seeds = make_seeds(10, base=0)
        assert len(set(seeds)) == 10

    def test_zero_replicates(self):
        seeds = make_seeds(0, base=0)
        assert seeds == []

    def test_one_replicate(self):
        seeds = make_seeds(1, base=0)
        assert len(seeds) == 1


class TestSeedEverything:
    def test_numpy_deterministic(self):
        seed_everything(123)
        a = np.random.rand(5)
        seed_everything(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_random_deterministic(self):
        seed_everything(999)
        a = [random.random() for _ in range(5)]
        seed_everything(999)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_give_different_results(self):
        seed_everything(1)
        a = np.random.rand(10)
        seed_everything(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)


class TestRunnerDeterminism:
    def test_rejection_abc_same_seed_produces_same_rows(self, tmp_path):
        cfg = test_helpers.make_fast_runner_config(
            "gaussian_mean.json",
            methods=["rejection_abc"],
            inference_overrides={"max_simulations": 80, "k": 10},
            execution_overrides={"n_replicates": 2, "base_seed": 1},
            plots={},
        )
        config_path = test_helpers.write_config(tmp_path, "gaussian_determinism.json", cfg)

        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        dir_a.mkdir()
        dir_b.mkdir()

        test_helpers.run_runner_main("gaussian_mean_runner.py", config_path, dir_a)
        test_helpers.run_runner_main("gaussian_mean_runner.py", config_path, dir_b)

        csv_a = dir_a / "gaussian_mean" / "data" / "raw_results.csv"
        csv_b = dir_b / "gaussian_mean" / "data" / "raw_results.csv"
        assert csv_a.exists(), f"Run A did not produce CSV at {csv_a}"
        assert csv_b.exists(), f"Run B did not produce CSV at {csv_b}"

        key_cols = ["method", "replicate", "seed", "step", "loss"]

        def _rows_as_set(path):
            with open(path) as f:
                return {tuple(row[c] for c in key_cols) for row in csv.DictReader(f)}

        def _row_count(path):
            with open(path) as f:
                return sum(1 for _ in csv.DictReader(f))

        set_a = _rows_as_set(csv_a)
        set_b = _rows_as_set(csv_b)

        # Coverage: both runs must have completed both replicates of rejection_abc
        methods_a = {(row[0], row[1]) for row in set_a}
        assert methods_a == {("rejection_abc", "0"), ("rejection_abc", "1")}, (
            f"Run A did not cover both rejection_abc replicates: {methods_a}"
        )

        # Determinism core check (REPR-02, D-05):
        assert set_a == set_b, (
            f"rejection_abc with same seed produced different row sets. "
            f"A-only: {set_a - set_b}, B-only: {set_b - set_a}"
        )
        assert _row_count(csv_a) == _row_count(csv_b), (
            "Row counts differ between runs with same seed — non-determinism detected."
        )
