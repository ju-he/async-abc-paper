"""Shared pytest fixtures for the experiments test suite."""
import json
import sys
import os
from pathlib import Path

import pytest

# Make async_abc importable from the experiments/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.records import ParticleRecord

MINIMAL_CONFIG = {
    "experiment_name": "test_experiment",
    "benchmark": {
        "name": "gaussian_mean",
        "observed_data_seed": 42,
        "n_obs": 50,
    },
    "methods": ["async_propulate_abc"],
    "inference": {
        "max_simulations": 5000,
        "n_workers": 4,
        "k": 20,
        "tol_init": 10.0,
        "scheduler_type": "acceptance_rate",
        "perturbation_scale": 0.8,
    },
    "execution": {
        "n_replicates": 3,
        "base_seed": 0,
    },
    "plots": {
        "posterior": True,
    },
}


@pytest.fixture
def minimal_config():
    return dict(MINIMAL_CONFIG)


@pytest.fixture
def config_file(tmp_path, minimal_config):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(minimal_config))
    return p


@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def sbc_config_file(tmp_path):
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {
            "name": "gaussian_mean",
            "n_obs": 40,
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
        "sbc": {
            "n_trials": 4,
            "coverage_levels": [0.5, 0.9],
        },
        "plots": {
            "rank_histogram": True,
            "coverage_table": True,
        },
    }
    path = tmp_path / "sbc.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.fixture
def straggler_config_file(tmp_path):
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
    path = tmp_path / "straggler.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.fixture
def sample_records():
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=1,
            params={"mu": 2.0},
            loss=2.0,
            weight=1.0,
            tolerance=5.0,
            wall_time=0.2,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=10,
            params={"mu": 1.0},
            loss=1.0,
            weight=1.0,
            tolerance=2.5,
            wall_time=1.0,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=25,
            params={"mu": 0.5},
            loss=0.5,
            weight=0.5,
            tolerance=1.0,
            wall_time=2.0,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=50,
            params={"mu": 0.1},
            loss=0.1,
            weight=0.5,
            tolerance=0.5,
            wall_time=3.0,
        ),
    ]


@pytest.fixture
def abc_smc_records():
    return [
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=1,
            params={"mu": 0.8},
            loss=0.8,
            weight=0.5,
            tolerance=1.0,
            wall_time=1.2,
            worker_id="0",
            sim_start_time=0.0,
            sim_end_time=1.0,
            generation=0,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=2,
            params={"mu": 0.6},
            loss=0.6,
            weight=0.5,
            tolerance=1.0,
            wall_time=1.2,
            worker_id="1",
            sim_start_time=0.0,
            sim_end_time=1.2,
            generation=0,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=3,
            params={"mu": 0.3},
            loss=0.3,
            weight=0.5,
            tolerance=0.5,
            wall_time=2.6,
            worker_id="0",
            sim_start_time=1.2,
            sim_end_time=2.2,
            generation=1,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=4,
            params={"mu": 0.1},
            loss=0.1,
            weight=0.5,
            tolerance=0.5,
            wall_time=2.6,
            worker_id="1",
            sim_start_time=1.2,
            sim_end_time=2.6,
            generation=1,
        ),
    ]
