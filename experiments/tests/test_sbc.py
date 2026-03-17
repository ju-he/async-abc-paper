"""Tests for Phase 4: SBC analysis and runner."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from async_abc.analysis.sbc import compute_rank, empirical_coverage, sbc_ranks


EXPERIMENTS_DIR = Path(__file__).parent.parent
PYTHON = sys.executable


def test_compute_rank_true_below_all():
    assert compute_rank(np.array([1.0, 2.0, 3.0]), 0.0) == 0


def test_compute_rank_true_above_all():
    assert compute_rank(np.array([1.0, 2.0, 3.0]), 4.0) == 3


def test_compute_rank_middle():
    rank = compute_rank(np.array([0.0, 1.0, 2.0]), 1.5)
    assert rank == 2


def test_sbc_ranks_returns_dataframe():
    trials = [
        {"trial": i, "param": "mu", "posterior_samples": np.linspace(0, 1, 100), "true_value": 0.5}
        for i in range(10)
    ]
    df = sbc_ranks(trials)
    assert "rank" in df.columns
    assert "trial" in df.columns


def test_empirical_coverage_uniform_posterior():
    rng = np.random.default_rng(42)
    trials = [
        {
            "trial": i,
            "param": "mu",
            "posterior_samples": rng.uniform(0, 1, 100),
            "true_value": rng.uniform(0, 1),
        }
        for i in range(500)
    ]
    df = empirical_coverage(trials, coverage_levels=[0.5, 0.9])
    row_50 = df[df["coverage_level"] == 0.5]["empirical_coverage"].iloc[0]
    assert abs(row_50 - 0.5) < 0.05


def test_sbc_runner_test_mode(tmp_path, sbc_config_file):
    result = subprocess.run(
        [
            PYTHON,
            str(EXPERIMENTS_DIR / "scripts" / "sbc_runner.py"),
            "--config",
            str(sbc_config_file),
            "--output-dir",
            str(tmp_path),
            "--test",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"SBC runner failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert (tmp_path / "sbc" / "data" / "sbc_ranks.csv").exists()
    assert (tmp_path / "sbc" / "data" / "coverage.csv").exists()
