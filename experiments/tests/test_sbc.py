"""Tests for Phase 4: SBC analysis and runner."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers

from async_abc.analysis.sbc import compute_rank, empirical_coverage, sbc_ranks
from async_abc.benchmarks import make_benchmark
from async_abc.inference.method_registry import method_execution_mode_for_cfg
from async_abc.io.config import load_config
from async_abc.io.records import ParticleRecord


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


def test_sbc_runner_executes_methods_in_method_major_order(tmp_path, monkeypatch):
    cfg = test_helpers.make_fast_runner_config(
        "sbc.json",
        methods=["async_propulate_abc", "abc_smc_baseline"],
        inference_overrides={
            "max_simulations": 20,
            "n_workers": 4,
            "k": 5,
            "n_generations": 2,
            "parallel_backend": "mpi",
        },
        execution_overrides={"n_replicates": 1, "base_seed": 11},
        plots={"rank_histogram": False, "coverage_table": False},
        top_level_updates={"sbc": {"n_trials": 3, "coverage_levels": [0.5]}},
    )
    config_path = test_helpers.write_config(tmp_path, "sbc_method_major.json", cfg)
    module = test_helpers.import_runner_module("sbc_runner.py")

    calls = []

    def fake_run_method_distributed(method, simulate_fn, limits, inference_cfg, output_dir, replicate, seed):
        calls.append((method, replicate))
        return [
            ParticleRecord(
                method=method,
                replicate=replicate,
                seed=seed,
                step=1,
                params={"mu": float(replicate)},
                loss=0.1,
                weight=1.0,
                tolerance=0.5,
                wall_time=0.1,
            )
        ]

    monkeypatch.setattr(module, "run_method_distributed", fake_run_method_distributed)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    monkeypatch.setattr(module, "write_timing_comparison_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "write_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "is_root_rank", lambda: True)
    monkeypatch.setattr(module, "allgather", lambda value: [value])
    monkeypatch.setattr(module, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "get_world_size", lambda: 4)

    module.main([
        "--config",
        str(config_path),
        "--output-dir",
        str(tmp_path),
    ])

    assert calls == [
        ("async_propulate_abc", 0),
        ("async_propulate_abc", 1),
        ("async_propulate_abc", 2),
        ("abc_smc_baseline", 0),
        ("abc_smc_baseline", 1),
        ("abc_smc_baseline", 2),
    ]


def test_sbc_full_config_treats_abc_smc_baseline_as_all_ranks_under_mpi():
    cfg = load_config(EXPERIMENTS_DIR / "configs" / "sbc.json", test_mode=False, small_mode=False)
    benchmark = make_benchmark(cfg["benchmark"])

    assert method_execution_mode_for_cfg("abc_smc_baseline", cfg["inference"], benchmark.simulate) == "all_ranks"


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
