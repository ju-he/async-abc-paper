"""Shared pytest fixtures for the experiments test suite."""
import json
import sys
import os
from pathlib import Path

import pytest

# Make async_abc importable from the experiments/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

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
