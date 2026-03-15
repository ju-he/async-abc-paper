"""Tests for async_abc.utils.seeding."""
import random

import numpy as np
import pytest

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
