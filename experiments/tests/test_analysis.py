"""Tests for async_abc.analysis.*"""
import numpy as np

from async_abc.analysis import (
    barrier_overhead_fraction,
    compute_ess,
    ess_over_time,
    generation_spans,
    loss_over_steps,
    posterior_quality_curve,
    time_to_threshold,
    tolerance_over_wall_time,
    wasserstein_at_checkpoints,
)
from async_abc.io.records import ParticleRecord


def _async_out_of_order_records():
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=1,
            params={"mu": 4.0},
            loss=4.0,
            tolerance=None,
            wall_time=0.30,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=1,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=2,
            params={"mu": 0.2},
            loss=0.2,
            tolerance=1.0,
            wall_time=0.20,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=2,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=3,
            params={"mu": 0.1},
            loss=0.1,
            tolerance=0.5,
            wall_time=0.10,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=3,
        ),
    ]


def _sync_generation_records():
    return [
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=1,
            params={"mu": 0.8},
            loss=0.8,
            tolerance=1.0,
            wall_time=1.0,
            generation=0,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=5,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=2,
            params={"mu": 0.7},
            loss=0.7,
            tolerance=1.0,
            wall_time=1.0,
            generation=0,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=5,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=3,
            params={"mu": 0.3},
            loss=0.3,
            tolerance=0.5,
            wall_time=2.0,
            generation=1,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=11,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=4,
            params={"mu": 0.2},
            loss=0.2,
            tolerance=0.5,
            wall_time=2.0,
            generation=1,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=11,
        ),
    ]


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


def test_posterior_quality_curve_wall_time(sample_records):
    df = posterior_quality_curve(
        sample_records,
        true_params={"mu": 0.0},
        axis_kind="wall_time",
        checkpoint_strategy="all",
        archive_size=20,
    )
    assert set(
        [
            "method",
            "replicate",
            "axis_kind",
            "axis_value",
            "checkpoint_id",
            "n_particles_used",
            "attempt_count",
            "time_semantics",
            "record_kind",
            "state_kind",
            "wall_time",
            "posterior_samples",
            "wasserstein",
        ]
    ) <= set(df.columns)
    assert (df["axis_kind"] == "wall_time").all()


def test_async_curve_uses_wall_time_not_raw_step_order():
    df = posterior_quality_curve(
        _async_out_of_order_records(),
        true_params={"mu": 0.0},
        axis_kind="wall_time",
        checkpoint_strategy="all",
        archive_size=5,
    )
    assert df["wall_time"].tolist() == sorted(df["wall_time"].tolist())
    assert df["attempt_count"].tolist() == [3, 2]


def test_async_curve_excludes_prior_phase_records():
    df = posterior_quality_curve(
        _async_out_of_order_records(),
        true_params={"mu": 0.0},
        axis_kind="posterior_samples",
        checkpoint_strategy="all",
        archive_size=5,
    )
    assert len(df) == 2
    assert (df["n_particles_used"] >= 1).all()


def test_sync_curve_uses_generation_snapshots():
    df = posterior_quality_curve(
        _sync_generation_records(),
        true_params={"mu": 0.0},
        axis_kind="attempt_budget",
        checkpoint_strategy="all",
    )
    assert len(df) == 2
    assert df["wall_time"].tolist() == [1.0, 2.0]
    assert df["attempt_count"].tolist() == [5, 11]
    assert (df["state_kind"] == "generation_population").all()


def test_wasserstein_at_checkpoints_uses_observable_states():
    df = wasserstein_at_checkpoints(
        _sync_generation_records(),
        true_params={"mu": 0.0},
        checkpoint_steps=[5, 10, 11],
    )
    assert df["step"].tolist() == [5, 10, 11]
    assert df["wall_time"].tolist() == [1.0, 1.0, 2.0]


def test_time_to_threshold_returns_none_for_impossible(sample_records):
    df = time_to_threshold(
        sample_records,
        true_params={"mu": 0.0},
        target_wasserstein=1e-9,
        archive_size=20,
    )
    assert df["wall_time_to_threshold"].isna().any()


def test_time_to_threshold_supports_attempt_budget(sample_records):
    df = time_to_threshold(
        sample_records,
        true_params={"mu": 0.0},
        target_wasserstein=10.0,
        axis_kind="attempt_budget",
        archive_size=20,
    )
    assert "attempts_to_threshold" in df.columns
    assert (df["attempts_to_threshold"] >= 0).all()


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


def test_posterior_quality_curve_multiparameter():
    """2-parameter posterior exercises the sliced-Wasserstein code path."""
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
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=i + 1,
        )
        for i in range(20)
    ]
    df = posterior_quality_curve(
        records,
        true_params={"mu": 0.0, "sigma": 1.0},
        axis_kind="attempt_budget",
        checkpoint_strategy="quantile",
        checkpoint_count=2,
        archive_size=20,
    )
    assert {"method", "replicate", "axis_value", "wall_time", "wasserstein"} <= set(df.columns)
    assert len(df) == 2
    assert (df["wasserstein"] >= 0.0).all()
