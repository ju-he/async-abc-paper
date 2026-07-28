"""Tests for async_abc.analysis.*"""
import numpy as np
import pytest

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


def test_time_to_threshold_respects_min_particles(sample_records):
    unrestricted = time_to_threshold(
        sample_records,
        true_params={"mu": 0.0},
        target_wasserstein=10.0,
        archive_size=20,
        min_particles=1,
    )
    guarded = time_to_threshold(
        sample_records,
        true_params={"mu": 0.0},
        target_wasserstein=10.0,
        archive_size=20,
        min_particles=1000,
    )
    assert not unrestricted.empty
    assert guarded["wall_time_to_threshold"].isna().all()


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


def _make_async_records(n_events, time_span):
    """Create n_events async records spread over time_span seconds."""
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=i + 1,
            params={"mu": float(i) * 0.05},
            loss=float(i) * 0.1,
            tolerance=5.0,
            wall_time=time_span * (i + 1) / n_events,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=i + 1,
        )
        for i in range(n_events)
    ]


def _make_sync_records(n_generations, particles_per_gen, time_span):
    """Create sync records with n_generations spread over time_span."""
    records = []
    step = 0
    for gen in range(n_generations):
        gen_time = time_span * (gen + 1) / n_generations
        for p in range(particles_per_gen):
            step += 1
            records.append(
                ParticleRecord(
                    method="abc_smc_baseline",
                    replicate=0,
                    seed=2,
                    step=step,
                    params={"mu": float(gen) * 0.1 + float(p) * 0.01},
                    loss=float(gen) * 0.1 + float(p) * 0.01,
                    tolerance=5.0 - gen * 0.5,
                    wall_time=gen_time,
                    generation=gen,
                    record_kind="population_particle",
                    time_semantics="generation_end",
                    attempt_count=(gen + 1) * 10,
                )
            )
    return records


def test_time_uniform_checkpoints_equal_density_across_methods():
    """With checkpoint_mode='time_uniform', both methods are resampled onto the same time grid."""
    time_span = 10.0
    async_records = _make_async_records(100, time_span)
    sync_records = _make_sync_records(5, 10, time_span)
    all_records = async_records + sync_records

    df = posterior_quality_curve(
        all_records,
        true_params={"mu": 0.0},
        axis_kind="wall_time",
        checkpoint_strategy="time_uniform",
        checkpoint_count=20,
    )

    async_df = df[df["method"] == "async_propulate_abc"]
    sync_df = df[df["method"] == "abc_smc_baseline"]

    # Async has events from t≈0.1 so should get all 20 grid points
    assert len(async_df) == 20, (
        f"async: expected 20 checkpoints, got {len(async_df)}"
    )
    # Sync has only 5 generation endpoints, so LOCF fills between them.
    # Grid points before the first generation are skipped.
    assert len(sync_df) >= 10, (
        f"sync: expected >= 10 checkpoints (LOCF), got {len(sync_df)}"
    )
    # Both should have many more checkpoints than native sync (5 generations)
    assert len(sync_df) > 5


def test_time_uniform_sync_uses_locf():
    """Between generations, the sync state should use the previous generation's population (LOCF)."""
    sync_records = _make_sync_records(2, 3, 10.0)  # gen0 at t=5, gen1 at t=10

    df = posterior_quality_curve(
        sync_records,
        true_params={"mu": 0.0},
        axis_kind="wall_time",
        checkpoint_strategy="time_uniform",
        checkpoint_count=10,
    )

    # Checkpoints before t=5 should have no data (or be absent)
    # Checkpoints between t=5 and t=10 should use gen0's population (LOCF)
    mid_checkpoints = df[(df["wall_time"] > 5.0) & (df["wall_time"] < 10.0)]
    if not mid_checkpoints.empty:
        # LOCF: particles from gen0 (3 particles)
        assert (mid_checkpoints["n_particles_used"] == 3).all()


def test_time_uniform_async_subset_of_full():
    """Time-uniform checkpoints should produce a subset of the all-events reconstruction."""
    records = _make_async_records(50, 5.0)

    df_all = posterior_quality_curve(
        records,
        true_params={"mu": 0.0},
        axis_kind="wall_time",
        checkpoint_strategy="all",
    )
    df_uniform = posterior_quality_curve(
        records,
        true_params={"mu": 0.0},
        axis_kind="wall_time",
        checkpoint_strategy="time_uniform",
        checkpoint_count=10,
    )

    assert len(df_uniform) <= len(df_all)
    # Wasserstein values at matching times should be identical
    for _, row in df_uniform.iterrows():
        matching = df_all[np.isclose(df_all["wall_time"], row["wall_time"], atol=1e-9)]
        if not matching.empty:
            assert np.isclose(
                row["wasserstein"], matching.iloc[0]["wasserstein"], rtol=1e-6
            )


def test_quality_row_includes_state_kind():
    """state_kind column must be present and correct for each method type."""
    async_records = _make_async_records(10, 2.0)
    sync_records = _make_sync_records(2, 3, 2.0)
    all_records = async_records + sync_records

    df = posterior_quality_curve(
        all_records,
        true_params={"mu": 0.0},
        axis_kind="wall_time",
        checkpoint_strategy="all",
    )

    assert "state_kind" in df.columns
    async_kinds = df[df["method"] == "async_propulate_abc"]["state_kind"].unique()
    sync_kinds = df[df["method"] == "abc_smc_baseline"]["state_kind"].unique()
    assert "archive_reconstruction" in async_kinds
    assert "generation_population" in sync_kinds


def test_wasserstein_documented_in_quality_curve():
    """The module docstring should describe the Wasserstein metric semantics."""
    from async_abc.analysis import convergence
    assert "wasserstein" in convergence.__doc__.lower()
    assert "point mass" in convergence.__doc__.lower()


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


def test_posterior_quality_curve_checkpoint_strategy_parameter():
    """All valid checkpoint_strategy values are accepted; invalid ones raise ValueError."""
    records = _make_async_records(20, 5.0)
    true_params = {"mu": 0.0}
    common_kwargs = dict(
        records=records,
        true_params=true_params,
        axis_kind="wall_time",
        archive_size=20,
    )

    # "all" — default, returns all checkpoints
    df_all = posterior_quality_curve(**common_kwargs, checkpoint_strategy="all")
    assert len(df_all) > 0

    # "time_uniform" — resamples onto shared grid
    df_tu = posterior_quality_curve(
        **common_kwargs, checkpoint_strategy="time_uniform", checkpoint_count=5,
    )
    assert len(df_tu) > 0

    # "quantile" — subsamples at quantile positions
    df_q = posterior_quality_curve(
        **common_kwargs, checkpoint_strategy="quantile", checkpoint_count=3,
    )
    assert len(df_q) > 0

    # Invalid strategy raises ValueError
    with pytest.raises(ValueError, match="Unsupported checkpoint_strategy"):
        posterior_quality_curve(**common_kwargs, checkpoint_strategy="bogus")


# ===========================================================================
# W3.3 — simulation_time_bias_report
# ===========================================================================


def _bias_test_records(*, slow_high_loss: bool, n: int = 200):
    """Synthetic records where sim_time correlates (or not) with loss.

    When slow_high_loss=True, slow simulations correlate with high losses
    (a strong bias signal). When False, sim_time and loss are independent.
    """
    rng = np.random.default_rng(0 if slow_high_loss else 1)
    records = []
    for i in range(n):
        if slow_high_loss:
            # loss in [0, 1]; sim_time proportional to loss → high correlation.
            loss = float(rng.uniform(0.0, 1.0))
            base = 0.01 + 0.5 * loss
        else:
            loss = float(rng.uniform(0.0, 1.0))
            base = float(rng.uniform(0.01, 0.5))
        sim_jitter = float(rng.normal(0.0, 0.005))
        sim_time = max(1e-6, base + sim_jitter)
        records.append(
            ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=42,
                step=i + 1,
                params={"x": loss},  # arbitrary
                loss=loss,
                weight=1.0,
                tolerance=0.5,  # accepted iff loss < 0.5
                wall_time=float(i),
                sim_start_time=float(i),
                sim_end_time=float(i) + sim_time,
                record_kind="accepted_particle",
                time_semantics="event_end",
            )
        )
    return records


def test_simulation_time_bias_report_detects_bias():
    from async_abc.analysis import simulation_time_bias_report

    report = simulation_time_bias_report(_bias_test_records(slow_high_loss=True))
    assert report["skip_reason"] is None
    assert report["n_records"] == 200
    # Bias direction: slow sims = high loss = REJECTED → mean_sim_time_rejected > mean_sim_time_accepted.
    assert report["mean_sim_time_rejected"] > report["mean_sim_time_accepted"]
    # Pearson correlation is positive and large under the bias model.
    assert report["pearson_corr_simtime_loss"] > 0.5
    # χ² should reject independence.
    assert report["chi2_p_value"] < 0.01


def test_simulation_time_bias_report_null_case():
    from async_abc.analysis import simulation_time_bias_report

    report = simulation_time_bias_report(_bias_test_records(slow_high_loss=False))
    assert report["skip_reason"] is None
    # Pearson correlation should be near zero under the null.
    assert abs(report["pearson_corr_simtime_loss"]) < 0.2
    # χ² independence p-value should be larger than under the strong-bias case.
    assert report["chi2_p_value"] > 0.05


def test_simulation_time_bias_report_empty_records():
    from async_abc.analysis import simulation_time_bias_report

    report = simulation_time_bias_report([])
    assert report["n_records"] == 0
    assert report["skip_reason"] == "empty_records"
    assert np.isnan(report["mean_sim_time_accepted"])


def test_simulation_time_bias_report_skips_when_sim_times_missing():
    """Records without sim_start/sim_end (sync methods) produce a skip reason."""
    from async_abc.analysis import simulation_time_bias_report

    records = [
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=42,
            step=i + 1,
            params={"x": 0.1},
            loss=0.1,
            weight=1.0,
            tolerance=0.5,
            wall_time=float(i),
            sim_start_time=None,
            sim_end_time=None,
            record_kind="accepted_particle",
            time_semantics="event_end",
        )
        for i in range(10)
    ]
    report = simulation_time_bias_report(records)
    assert report["skip_reason"] == "no_finite_sim_time_or_loss"
    assert np.isnan(report["mean_sim_time_accepted"])


# ===========================================================================
# W3.2 — ess_vs_n_at_fixed_S
# ===========================================================================


def _uniform_weight_records(n: int):
    """Records with all weights equal to 1 → relative ESS should be 1.0."""
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=i + 1,
            params={"x": 0.5},
            loss=0.1,
            weight=1.0,
            tolerance=1.0,
            wall_time=float(i),
            record_kind="accepted_particle",
            time_semantics="event_end",
        )
        for i in range(n)
    ]


def _heavily_skewed_weight_records(n: int):
    """One huge weight + many tiny ones → relative ESS → 0 (degenerate)."""
    weights = [1e-3] * (n - 1) + [1e3]
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=i + 1,
            params={"x": 0.5},
            loss=0.1,
            weight=float(w),
            tolerance=1.0,
            wall_time=float(i),
            record_kind="accepted_particle",
            time_semantics="event_end",
        )
        for i, w in enumerate(weights)
    ]


def test_ess_vs_n_at_fixed_S_uniform_returns_one():
    from async_abc.analysis import ess_vs_n_at_fixed_S

    df = ess_vs_n_at_fixed_S(_uniform_weight_records(200), window=50)
    assert not df.empty
    # All sliding windows over equal weights have relative ESS = 1.0.
    assert np.allclose(df["relative_ess"].to_numpy(), 1.0)


def test_ess_vs_n_at_fixed_S_degenerate_low_when_skewed():
    from async_abc.analysis import ess_vs_n_at_fixed_S

    df = ess_vs_n_at_fixed_S(_heavily_skewed_weight_records(100), window=50)
    # The last sliding window contains the giant weight, dominating the rest.
    last = df[df["n"] == df["n"].max()]
    assert float(last["relative_ess"].iloc[0]) < 0.5


def test_ess_vs_n_at_fixed_S_empty_returns_empty_with_schema():
    from async_abc.analysis import ess_vs_n_at_fixed_S

    df = ess_vs_n_at_fixed_S([], window=50)
    assert df.empty
    assert set(df.columns) == {"method", "replicate", "n", "ess", "relative_ess"}


def test_ess_vs_n_at_fixed_S_filters_by_method():
    from async_abc.analysis import ess_vs_n_at_fixed_S

    records = _uniform_weight_records(20) + [
        ParticleRecord(
            method="pyabc_smc",
            replicate=0,
            seed=2,
            step=i + 1,
            params={"x": 0.5},
            loss=0.1,
            weight=1.0,
            tolerance=1.0,
            wall_time=float(i),
            record_kind="accepted_particle",
            time_semantics="event_end",
        )
        for i in range(20)
    ]
    df = ess_vs_n_at_fixed_S(records, window=10, method_label="async_propulate_abc")
    assert set(df["method"].unique()) == {"async_propulate_abc"}


def _tightening_tolerance_records(n: int = 60):
    """Async records whose bandwidth tightens below every achieved loss.

    Mirrors a real smooth-kernel run: the scheduler drives ``tolerance`` down
    monotonically past the best loss ever achieved. Under the *hard*-kernel
    archive rule no particle then satisfies ``loss < eps``, so the quality curve
    silently stops; under a smooth kernel every particle stays archive-eligible.
    """
    records = []
    for i in range(n):
        # loss floors at 0.5; tolerance decays past it and ends well below.
        loss = 0.5 + 1.0 / (i + 1)
        tolerance = 2.0 * (0.9**i)
        records.append(
            ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=1,
                step=i + 1,
                params={"mu": loss},
                loss=loss,
                weight=1.0,
                tolerance=tolerance,
                wall_time=float(i),
                record_kind="simulation_attempt",
                time_semantics="event_end",
                attempt_count=i + 1,
            )
        )
    return records


def test_smooth_kernel_quality_curve_spans_full_run_when_eps_drops_below_all_losses():
    """Regression: a smooth-kernel curve must not truncate when eps < min(loss).

    The archive-reconstruction rule is kernel-dependent (it mirrors
    ``ABCPMC._reconstruct_archive``). Applying the hard rule to a smooth-kernel
    run silently dropped every checkpoint after eps fell below the best loss,
    which truncated the Cellular Potts curve at ~28% of its budget.
    """
    records = _tightening_tolerance_records()
    last_wall_time = max(r.wall_time for r in records)

    smooth = posterior_quality_curve(
        records,
        true_params={"mu": 0.5},
        axis_kind="wall_time",
        checkpoint_strategy="all",
        archive_size=10,
        kernel="gaussian",
    )
    hard = posterior_quality_curve(
        records,
        true_params={"mu": 0.5},
        axis_kind="wall_time",
        checkpoint_strategy="all",
        archive_size=10,
        kernel="hard",
    )

    # The smooth curve reaches the end of the run...
    assert smooth["axis_value"].max() == pytest.approx(last_wall_time)
    # ...while the hard rule stops early on the very same records.
    assert hard["axis_value"].max() < last_wall_time
    assert len(smooth) > len(hard)
