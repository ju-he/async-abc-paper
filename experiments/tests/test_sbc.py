"""Tests for Phase 4: SBC analysis and runner."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers

from async_abc.analysis.sbc import compute_rank, compute_rank_weighted, empirical_coverage, sbc_ranks
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
    assert int(df[df["coverage_level"] == 0.5]["n_trials"].iloc[0]) == 500


def test_sbc_plot_metadata_is_complete(tmp_path):
    module = test_helpers.import_runner_module("sbc_runner.py")
    from async_abc.io.paths import OutputDir

    output_dir = OutputDir(tmp_path, "sbc").ensure()
    ranks_df = pd.DataFrame(
        {
            "method": ["async_propulate_abc", "async_propulate_abc"],
            "param": ["mu", "mu"],
            "trial": [0, 1],
            "rank": [1, 2],
            "n_samples": [5, 5],
        }
    )
    coverage_df = pd.DataFrame(
        {
            "method": ["async_propulate_abc"],
            "param": ["mu"],
            "coverage_level": [0.5],
            "empirical_coverage": [0.6],
            "n_trials": [2],
        }
    )
    module._plot_rank_histogram(ranks_df, output_dir)
    module._plot_coverage_table(coverage_df, output_dir)
    rank_meta = json.loads((output_dir.plots / "rank_histogram_meta.json").read_text())
    coverage_meta = json.loads((output_dir.plots / "coverage_table_meta.json").read_text())
    assert rank_meta["plot_name"] == "rank_histogram"
    assert rank_meta["summary_plot"] is True
    assert rank_meta["benchmark"] is False
    assert coverage_meta["plot_name"] == "coverage_table"
    assert coverage_meta["summary_plot"] is True
    assert coverage_meta["benchmark"] is False


def test_sbc_runner_and_finalizer_plot_outputs_match(tmp_path):
    runner_module = test_helpers.import_runner_module("sbc_runner.py")
    from async_abc.io.paths import OutputDir
    from async_abc.utils import shard_finalizers

    runner_output = OutputDir(tmp_path / "runner", "sbc").ensure()
    finalizer_output = OutputDir(tmp_path / "finalizer", "sbc").ensure()
    ranks_df = pd.DataFrame(
        {
            "method": ["async_propulate_abc", "async_propulate_abc"],
            "param": ["mu", "mu"],
            "trial": [0, 1],
            "rank": [1, 2],
            "n_samples": [5, 5],
        }
    )
    coverage_df = pd.DataFrame(
        {
            "method": ["async_propulate_abc"],
            "param": ["mu"],
            "coverage_level": [0.5],
            "empirical_coverage": [0.6],
            "n_trials": [2],
        }
    )

    runner_module._plot_rank_histogram(ranks_df, runner_output)
    runner_module._plot_coverage_table(coverage_df, runner_output)
    shard_finalizers._plot_rank_histogram(ranks_df, finalizer_output)
    shard_finalizers._plot_coverage_table(coverage_df, finalizer_output)

    runner_rank_meta = json.loads((runner_output.plots / "rank_histogram_meta.json").read_text())
    finalizer_rank_meta = json.loads((finalizer_output.plots / "rank_histogram_meta.json").read_text())
    runner_coverage_meta = json.loads((runner_output.plots / "coverage_table_meta.json").read_text())
    finalizer_coverage_meta = json.loads((finalizer_output.plots / "coverage_table_meta.json").read_text())

    stable_keys = {"plot_name", "title", "summary_plot", "diagnostic_plot", "experiment_name", "benchmark", "methods"}
    assert {key: runner_rank_meta[key] for key in stable_keys} == {
        key: finalizer_rank_meta[key] for key in stable_keys
    }
    assert {key: runner_coverage_meta[key] for key in stable_keys} == {
        key: finalizer_coverage_meta[key] for key in stable_keys
    }


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
        # Replace (not update) the sbc block so the benchmarks list from sbc.json
        # is not inherited — this test only checks method-major ordering, not multi-benchmark.
        replace_top_level={"sbc": {"n_trials": 3, "coverage_levels": [0.5]}},
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


def test_posterior_samples_reconstructs_async_final_archive_per_replicate():
    module = test_helpers.import_runner_module("sbc_runner.py")
    records = [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=1,
            params={"mu": 4.5},
            loss=0.45,
            tolerance=3.0,
            wall_time=1.0,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=2,
            params={"mu": 3.5},
            loss=0.35,
            tolerance=2.0,
            wall_time=2.0,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=3,
            params={"mu": 2.5},
            loss=0.25,
            tolerance=1.0,
            wall_time=3.0,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=4,
            params={"mu": 1.5},
            loss=0.15,
            tolerance=0.5,
            wall_time=4.0,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=1,
            seed=2,
            step=1,
            params={"mu": -4.0},
            loss=0.40,
            tolerance=2.0,
            wall_time=1.0,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=1,
            seed=2,
            step=2,
            params={"mu": -3.0},
            loss=0.30,
            tolerance=0.5,
            wall_time=2.0,
        ),
    ]
    samples, weights = module._posterior_samples(records, "mu", archive_size=2)
    assert np.allclose(np.sort(samples), np.array([-4.0, -3.0, 1.5, 2.5]))


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


# ── Phase 1: Weighted SBC ranks and coverage ─────────────────────────────────


def test_compute_rank_weighted_uniform_weights_matches_unweighted():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, 100)
    true_value = 0.3
    unweighted = compute_rank(samples, true_value)
    weights = np.ones(len(samples))
    # With uniform weights and large enough sample, resampled rank is within ±5
    results = [compute_rank_weighted(samples, weights, true_value, seed=i) for i in range(20)]
    assert all(abs(r - unweighted) <= 10 for r in results), (
        f"Weighted rank with uniform weights too far from unweighted: {results} vs {unweighted}"
    )


def test_compute_rank_weighted_concentrated_weight():
    # One particle at -5.0 gets all weight; true_value=0.0 → rank should be 1 (above the one particle)
    samples = np.array([-5.0, 0.5, 1.0, 2.0, 3.0])
    weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    true_value = 0.0
    # After resampling, all samples are -5.0; true_value=0.0 is above all → rank = len(samples)
    for seed in range(5):
        rank = compute_rank_weighted(samples, weights, true_value, seed=seed)
        assert rank == len(samples), f"Expected rank {len(samples)}, got {rank}"


def test_compute_rank_weighted_none_weights_fallback():
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    # None weights should fall back to unweighted compute_rank
    result = compute_rank_weighted(samples, None, 1.5)
    assert result == compute_rank(samples, 1.5)


def test_compute_rank_weighted_zero_weights_fallback():
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    weights = np.zeros(len(samples))
    result = compute_rank_weighted(samples, weights, 1.5)
    assert result == compute_rank(samples, 1.5)


def test_sbc_ranks_uses_posterior_weights_when_present():
    # Construct a trial where weights are concentrated at one extreme
    # so weighted rank ≠ unweighted rank with high probability
    samples = np.linspace(0, 1, 50)
    # All weight on sample at 0.0 → every true_value > 0 gets rank = 50
    weights = np.zeros(50)
    weights[0] = 1.0
    trials = [{"trial": 0, "method": "m", "param": "x", "true_value": 0.5,
                "posterior_samples": samples, "posterior_weights": weights}]
    df = sbc_ranks(trials)
    assert df["rank"].iloc[0] == 50


def test_sbc_ranks_falls_back_without_posterior_weights():
    samples = np.linspace(0, 1, 20)
    trials = [{"trial": 0, "method": "m", "param": "x",
               "true_value": 0.5, "posterior_samples": samples}]
    df_no_weights = sbc_ranks(trials)
    expected = compute_rank(samples, 0.5)
    assert df_no_weights["rank"].iloc[0] == expected


def test_empirical_coverage_weighted_calibrated_posterior():
    # Weighted posterior that is uniform on [0,1]; true_value drawn from same → 90% coverage expected
    rng = np.random.default_rng(7)
    n = 500
    samples_per_trial = 100
    trials = []
    for i in range(n):
        samples = np.linspace(0, 1, samples_per_trial)
        # Uniform weights → identical to unweighted
        weights = np.ones(samples_per_trial)
        true_value = rng.uniform(0, 1)
        trials.append({"trial": i, "method": "m", "param": "x",
                        "true_value": true_value,
                        "posterior_samples": samples,
                        "posterior_weights": weights})
    df = empirical_coverage(trials, [0.9])
    row = df[df["coverage_level"] == 0.9]["empirical_coverage"].iloc[0]
    assert abs(row - 0.9) < 0.05, f"Expected ~0.9 empirical coverage, got {row}"


def test_trial_records_jsonl_roundtrip_with_weights(tmp_path):
    module = test_helpers.import_runner_module("sbc_runner.py")
    from async_abc.io.paths import OutputDir
    output_dir = OutputDir(tmp_path, "sbc").ensure()
    samples = np.array([0.1, 0.5, 0.9])
    weights = np.array([0.2, 0.5, 0.3])
    records = [{"trial": 0, "method": "m", "param": "x",
                "true_value": 0.4, "posterior_samples": samples,
                "posterior_weights": weights}]
    path = output_dir.data / "sbc_trials.jsonl"
    module._write_trial_records_jsonl(records, path)
    loaded = []
    with open(path) as f:
        for line in f:
            loaded.append(json.loads(line.strip()))
    assert "posterior_weights" in loaded[0]
    assert np.allclose(loaded[0]["posterior_weights"], weights.tolist())
    assert np.allclose(loaded[0]["posterior_samples"], samples.tolist())


def test_posterior_samples_returns_weights():
    module = test_helpers.import_runner_module("sbc_runner.py")
    records = [
        ParticleRecord(
            method="async_propulate_abc", replicate=0, seed=1, step=1,
            params={"mu": 0.5}, loss=0.1, weight=2.0, tolerance=0.5, wall_time=1.0,
        ),
        ParticleRecord(
            method="async_propulate_abc", replicate=0, seed=1, step=2,
            params={"mu": -0.5}, loss=0.05, weight=0.5, tolerance=0.5, wall_time=2.0,
        ),
    ]
    samples, weights = module._posterior_samples(records, "mu", archive_size=2)
    assert samples.shape == (2,)
    assert weights.shape == (2,)
    assert all(w > 0 for w in weights)


# ── Phase 2: Multi-benchmark SBC ─────────────────────────────────────────────


def test_sbc_config_benchmarks_list_parsed():
    module = test_helpers.import_runner_module("sbc_runner.py")
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {"name": "gaussian_mean"},
        "sbc": {
            "n_trials": 2,
            "coverage_levels": [0.5],
            "benchmarks": [
                {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                 "prior_low": -5.0, "prior_high": 5.0, "inference_overrides": {}},
                {"name": "gandk", "n_obs": 50, "inference_overrides": {"max_simulations": 200}},
            ],
        },
    }
    benchmarks = module._resolve_benchmark_configs(cfg)
    names = [b["name"] for b in benchmarks]
    assert "gaussian_mean" in names
    assert "gandk" in names
    assert len(benchmarks) == 2


def test_sbc_config_no_benchmarks_key_falls_back_to_top_level():
    module = test_helpers.import_runner_module("sbc_runner.py")
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                      "prior_low": -5.0, "prior_high": 5.0},
        "sbc": {"n_trials": 2, "coverage_levels": [0.5]},
    }
    benchmarks = module._resolve_benchmark_configs(cfg)
    assert len(benchmarks) == 1
    assert benchmarks[0]["name"] == "gaussian_mean"


def test_sbc_ranks_csv_has_benchmark_column(tmp_path, monkeypatch):
    module = test_helpers.import_runner_module("sbc_runner.py")
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                      "prior_low": -5.0, "prior_high": 5.0},
        "methods": ["rejection_abc"],
        "inference": {"max_simulations": 50, "n_workers": 1, "k": 5, "tol_init": 5.0,
                      "n_generations": 1, "scheduler_type": "acceptance_rate",
                      "perturbation_scale": 0.8},
        "execution": {"n_replicates": 1, "base_seed": 0},
        "sbc": {
            "n_trials": 2,
            "coverage_levels": [0.5],
            "benchmarks": [
                {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                 "prior_low": -5.0, "prior_high": 5.0, "inference_overrides": {}},
            ],
        },
        "plots": {"rank_histogram": False, "coverage_table": False},
    }
    config_path = test_helpers.write_config(tmp_path, "sbc_bench.json", cfg)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    monkeypatch.setattr(module, "write_timing_comparison_csv", lambda *_a, **_kw: None)
    monkeypatch.setattr(module, "write_metadata", lambda *_a, **_kw: None)
    monkeypatch.setattr(module, "is_root_rank", lambda: True)
    monkeypatch.setattr(module, "allgather", lambda v: [v])
    monkeypatch.setattr(module, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "get_world_size", lambda: 1)
    module.main(["--config", str(config_path), "--output-dir", str(tmp_path)])
    ranks_df = pd.read_csv(tmp_path / "sbc" / "data" / "sbc_ranks.csv")
    assert "benchmark" in ranks_df.columns


def test_sbc_coverage_csv_has_benchmark_column(tmp_path, monkeypatch):
    module = test_helpers.import_runner_module("sbc_runner.py")
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                      "prior_low": -5.0, "prior_high": 5.0},
        "methods": ["rejection_abc"],
        "inference": {"max_simulations": 50, "n_workers": 1, "k": 5, "tol_init": 5.0,
                      "n_generations": 1, "scheduler_type": "acceptance_rate",
                      "perturbation_scale": 0.8},
        "execution": {"n_replicates": 1, "base_seed": 0},
        "sbc": {
            "n_trials": 2,
            "coverage_levels": [0.5],
            "benchmarks": [
                {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                 "prior_low": -5.0, "prior_high": 5.0, "inference_overrides": {}},
            ],
        },
        "plots": {"rank_histogram": False, "coverage_table": False},
    }
    config_path = test_helpers.write_config(tmp_path, "sbc_bench2.json", cfg)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    monkeypatch.setattr(module, "write_timing_comparison_csv", lambda *_a, **_kw: None)
    monkeypatch.setattr(module, "write_metadata", lambda *_a, **_kw: None)
    monkeypatch.setattr(module, "is_root_rank", lambda: True)
    monkeypatch.setattr(module, "allgather", lambda v: [v])
    monkeypatch.setattr(module, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "get_world_size", lambda: 1)
    module.main(["--config", str(config_path), "--output-dir", str(tmp_path)])
    cov_df = pd.read_csv(tmp_path / "sbc" / "data" / "coverage.csv")
    assert "benchmark" in cov_df.columns


def test_sbc_trial_records_jsonl_has_benchmark_field(tmp_path, monkeypatch):
    module = test_helpers.import_runner_module("sbc_runner.py")
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                      "prior_low": -5.0, "prior_high": 5.0},
        "methods": ["rejection_abc"],
        "inference": {"max_simulations": 50, "n_workers": 1, "k": 5, "tol_init": 5.0,
                      "n_generations": 1, "scheduler_type": "acceptance_rate",
                      "perturbation_scale": 0.8},
        "execution": {"n_replicates": 1, "base_seed": 0},
        "sbc": {
            "n_trials": 2,
            "coverage_levels": [0.5],
            "benchmarks": [
                {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                 "prior_low": -5.0, "prior_high": 5.0, "inference_overrides": {}},
            ],
        },
        "plots": {"rank_histogram": False, "coverage_table": False},
    }
    config_path = test_helpers.write_config(tmp_path, "sbc_bench3.json", cfg)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    monkeypatch.setattr(module, "write_timing_comparison_csv", lambda *_a, **_kw: None)
    monkeypatch.setattr(module, "write_metadata", lambda *_a, **_kw: None)
    monkeypatch.setattr(module, "is_root_rank", lambda: True)
    monkeypatch.setattr(module, "allgather", lambda v: [v])
    monkeypatch.setattr(module, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "get_world_size", lambda: 1)
    module.main(["--config", str(config_path), "--output-dir", str(tmp_path)])
    jsonl_path = tmp_path / "sbc" / "data" / "sbc_trials.jsonl"
    with open(jsonl_path) as f:
        row = json.loads(f.readline())
    assert "benchmark" in row


def test_plot_rank_histogram_multibenchmark_facets(tmp_path):
    from async_abc.io.paths import OutputDir
    from async_abc.plotting.sbc import plot_rank_histogram
    output_dir = OutputDir(tmp_path, "sbc").ensure()
    ranks_df = pd.DataFrame({
        "benchmark": ["gaussian_mean", "gaussian_mean", "gandk", "gandk"],
        "method": ["m", "m", "m", "m"],
        "param": ["x", "x", "x", "x"],
        "trial": [0, 1, 0, 1],
        "rank": [1, 2, 1, 2],
        "n_samples": [5, 5, 5, 5],
    })
    plot_rank_histogram(ranks_df, output_dir)
    meta = json.loads((output_dir.plots / "rank_histogram_meta.json").read_text())
    # Should have one panel per (benchmark, method) combination = 2
    assert meta["n_panels"] == 2


def test_plot_coverage_table_multibenchmark_lines(tmp_path):
    from async_abc.io.paths import OutputDir
    from async_abc.plotting.sbc import plot_coverage_table
    output_dir = OutputDir(tmp_path, "sbc").ensure()
    coverage_df = pd.DataFrame({
        "benchmark": ["gaussian_mean", "gandk"],
        "method": ["m", "m"],
        "param": ["x", "x"],
        "coverage_level": [0.9, 0.9],
        "empirical_coverage": [0.88, 0.91],
        "n_trials": [10, 10],
    })
    plot_coverage_table(coverage_df, output_dir)
    meta = json.loads((output_dir.plots / "coverage_table_meta.json").read_text())
    assert len(meta["methods"]) == 2  # two (benchmark, method) combinations


# ── Phase 3: Histogram bins ───────────────────────────────────────────────────


def _capture_rank_histogram_axes(ranks_df, tmp_path):
    """Helper: call plot_rank_histogram and return the axes before figure is closed."""
    import async_abc.plotting.sbc as sbc_plot_mod
    captured = {}

    def fake_save_figure(fig, path_stem, data=None, metadata=None):
        captured["axes"] = list(fig.axes)
        import matplotlib.pyplot as plt
        plt.close(fig)

    from async_abc.io.paths import OutputDir
    output_dir = OutputDir(tmp_path, "sbc").ensure()
    orig = sbc_plot_mod.save_figure
    sbc_plot_mod.save_figure = fake_save_figure
    try:
        sbc_plot_mod.plot_rank_histogram(ranks_df, output_dir)
    finally:
        sbc_plot_mod.save_figure = orig
    return captured.get("axes", [])


def test_rank_histogram_bins_equals_n_samples_plus_one(tmp_path):
    n_samples = 50
    ranks_df = pd.DataFrame({
        "method": ["m"] * 10,
        "param": ["x"] * 10,
        "trial": list(range(10)),
        "rank": list(range(0, 50, 5)),
        "n_samples": [n_samples] * 10,
    })
    axes = _capture_rank_histogram_axes(ranks_df, tmp_path)
    assert axes, "No axes captured"
    patches = axes[0].patches
    assert len(patches) == n_samples + 1, f"Expected {n_samples + 1} bins, got {len(patches)}"


def test_rank_histogram_large_n_samples_no_cap(tmp_path):
    n_samples = 200
    ranks_df = pd.DataFrame({
        "method": ["m"] * 5,
        "param": ["x"] * 5,
        "trial": list(range(5)),
        "rank": [0, 50, 100, 150, 200],
        "n_samples": [n_samples] * 5,
    })
    axes = _capture_rank_histogram_axes(ranks_df, tmp_path)
    assert axes, "No axes captured"
    patches = axes[0].patches
    # Should be 201 bins, NOT capped at 30
    assert len(patches) == n_samples + 1, f"Expected {n_samples + 1} bins, got {len(patches)}"


def test_rank_histogram_uniform_reference_line_present(tmp_path):
    ranks_df = pd.DataFrame({
        "method": ["m"] * 5,
        "param": ["x"] * 5,
        "trial": list(range(5)),
        "rank": [0, 2, 4, 6, 8],
        "n_samples": [10] * 5,
    })
    axes = _capture_rank_histogram_axes(ranks_df, tmp_path)
    assert axes, "No axes captured"
    lines = axes[0].lines
    assert len(lines) >= 1, "Expected at least one reference line (axhline) in rank histogram"


# ── Phase 4: Trial dropout diagnostics ───────────────────────────────────────


def test_extend_trial_records_returns_false_on_empty_samples():
    module = test_helpers.import_runner_module("sbc_runner.py")
    target = []
    # records produce no archive particles → empty samples
    result = module._extend_trial_records(
        target=target,
        trial_idx=0,
        method="async_propulate_abc",
        benchmark_name="gaussian_mean",
        true_params={"mu": 0.5},
        param_names=["mu"],
        records=[],  # empty → no archive
        archive_size=5,
    )
    assert result is False
    assert len(target) == 0


def test_extend_trial_records_returns_true_on_success():
    module = test_helpers.import_runner_module("sbc_runner.py")
    records = [
        ParticleRecord(
            method="async_propulate_abc", replicate=0, seed=1, step=1,
            params={"mu": 0.5}, loss=0.05, weight=1.0, tolerance=0.5, wall_time=1.0,
        ),
    ]
    target = []
    result = module._extend_trial_records(
        target=target,
        trial_idx=0,
        method="async_propulate_abc",
        benchmark_name="gaussian_mean",
        true_params={"mu": 0.3},
        param_names=["mu"],
        records=records,
        archive_size=5,
    )
    assert result is True
    assert len(target) == 1


def test_extend_trial_records_logs_warning_on_dropout(caplog):
    import logging
    module = test_helpers.import_runner_module("sbc_runner.py")
    with caplog.at_level(logging.WARNING):
        module._extend_trial_records(
            target=[],
            trial_idx=7,
            method="async_propulate_abc",
            benchmark_name="gaussian_mean",
            true_params={"mu": 0.5},
            param_names=["mu"],
            records=[],
            archive_size=5,
        )
    assert any("trial" in rec.message.lower() or "dropout" in rec.message.lower()
               for rec in caplog.records), f"Expected dropout warning, got: {[r.message for r in caplog.records]}"


def test_metadata_includes_trial_dropout_counts(tmp_path):
    module = test_helpers.import_runner_module("sbc_runner.py")
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                      "prior_low": -5.0, "prior_high": 5.0},
        "methods": ["rejection_abc"],
        "inference": {"max_simulations": 50, "n_workers": 1, "k": 5, "tol_init": 5.0,
                      "n_generations": 1, "scheduler_type": "acceptance_rate",
                      "perturbation_scale": 0.8},
        "execution": {"n_replicates": 1, "base_seed": 0},
        "sbc": {
            "n_trials": 2,
            "coverage_levels": [0.5],
            "benchmarks": [
                {"name": "gaussian_mean", "n_obs": 40, "sigma_obs": 1.0,
                 "prior_low": -5.0, "prior_high": 5.0, "inference_overrides": {}},
            ],
        },
        "plots": {"rank_histogram": False, "coverage_table": False},
    }
    config_path = test_helpers.write_config(tmp_path, "sbc_dropout.json", cfg)

    captured_meta_extra = {}
    orig_write_metadata = module.write_metadata

    def fake_write_metadata(output_dir, cfg, extra=None):
        captured_meta_extra.update(extra or {})

    module.write_metadata = fake_write_metadata
    try:
        module.main(["--config", str(config_path), "--output-dir", str(tmp_path)])
    finally:
        module.write_metadata = orig_write_metadata

    assert "trial_dropouts" in captured_meta_extra, (
        f"Expected 'trial_dropouts' in metadata extra, got keys: {list(captured_meta_extra.keys())}"
    )
