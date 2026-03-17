"""Tests for async_abc.analysis.*"""
import numpy as np

from async_abc.analysis import (
    barrier_overhead_fraction,
    compute_ess,
    ess_over_time,
    generation_spans,
    loss_over_steps,
    time_to_threshold,
    tolerance_over_wall_time,
    wasserstein_at_checkpoints,
)


def test_compute_ess_uniform_weights():
    w = np.ones(10)
    assert abs(compute_ess(w) - 10.0) < 1e-6


def test_compute_ess_degenerate():
    w = np.zeros(10)
    w[0] = 1.0
    assert abs(compute_ess(w) - 1.0) < 1e-6


def test_ess_over_time_returns_dataframe(sample_records):
    df = ess_over_time(sample_records, method="async_propulate_abc")
    assert {"step", "ess"} <= set(df.columns)
    assert len(df) > 0


def test_wasserstein_at_checkpoints(sample_records):
    df = wasserstein_at_checkpoints(
        sample_records,
        true_params={"mu": 0.0},
        checkpoint_steps=[10, 50],
    )
    assert {"method", "replicate", "step", "wall_time", "wasserstein"} <= set(df.columns)
    assert len(df) == 2


def test_time_to_threshold_returns_none_for_impossible(sample_records):
    df = time_to_threshold(
        sample_records,
        true_params={"mu": 0.0},
        target_wasserstein=1e-9,
    )
    assert df["wall_time_to_threshold"].isna().any()


def test_tolerance_over_wall_time(sample_records):
    df = tolerance_over_wall_time(sample_records)
    assert "wall_time" in df.columns
    assert "tolerance" in df.columns


def test_loss_over_steps(sample_records):
    df = loss_over_steps(sample_records)
    assert {"method", "replicate", "step", "loss"} <= set(df.columns)


def test_generation_spans_requires_generation_field(abc_smc_records):
    df = generation_spans(abc_smc_records)
    assert "generation" in df.columns
    assert (df["gen_end"] >= df["gen_start"]).all()


def test_barrier_overhead_fraction_returns_fraction(abc_smc_records):
    df = barrier_overhead_fraction(abc_smc_records)
    assert {"method", "replicate", "barrier_overhead_fraction"} <= set(df.columns)
    assert (df["barrier_overhead_fraction"] >= 0.0).all()
    assert (df["barrier_overhead_fraction"] <= 1.0).all()


def test_wasserstein_at_checkpoints_multiparameter():
    """2-parameter posterior exercises the sliced-Wasserstein code path."""
    from async_abc.io.records import ParticleRecord

    records = [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=i + 1,
            params={"mu": float(i) * 0.1, "sigma": 1.0 + float(i) * 0.05},
            loss=float(i) * 0.1,
            weight=1.0,
            tolerance=5.0,
            wall_time=float(i) * 0.5,
        )
        for i in range(20)
    ]
    df = wasserstein_at_checkpoints(
        records,
        true_params={"mu": 0.0, "sigma": 1.0},
        checkpoint_steps=[10, 20],
    )
    assert {"method", "replicate", "step", "wall_time", "wasserstein"} <= set(df.columns)
    assert len(df) == 2
    assert (df["wasserstein"] >= 0.0).all()
