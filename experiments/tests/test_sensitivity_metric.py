"""Tests for compute_sensitivity_quality_summary().

All tests in this module are written before the implementation and are
expected to fail until analysis/sensitivity.py is created.
"""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

# The function under test — does not exist yet, so all tests fail on import.
from async_abc.analysis.sensitivity import compute_sensitivity_quality_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIELDNAMES = [
    "method", "replicate", "seed", "step",
    "param_mu",
    "loss", "weight", "tolerance", "wall_time",
    "worker_id", "sim_start_time", "sim_end_time",
    "generation", "record_kind", "time_semantics", "attempt_count",
]


def _write_variant_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal sensitivity variant CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_rows(
    *,
    n_steps: int,
    n_replicates: int,
    param_mu: float,
    method: str = "async_propulate_abc__k=50__perturbation_scale=0.4__tol_init_multiplier=1.0",
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate synthetic particle rows for ``n_replicates`` replicates."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for rep in range(n_replicates):
        for step in range(1, n_steps + 1):
            rows.append({
                "method": method,
                "replicate": rep,
                "seed": rep * 100,
                "step": step,
                "param_mu": param_mu + rng.normal(0, noise) if noise else param_mu,
                "loss": 0.1,
                "weight": 1.0,
                "tolerance": 1.0,
                "wall_time": float(step),
                "worker_id": "",
                "sim_start_time": "",
                "sim_end_time": "",
                "generation": "",
                "record_kind": "",
                "time_semantics": "",
                "attempt_count": step,
            })
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeSensitivityQualitySummary:
    """Basic contract tests for compute_sensitivity_quality_summary()."""

    def test_returns_dataframe_with_required_columns(self, tmp_path):
        """Result must have wasserstein_mean, wasserstein_std, n_replicates and grid key columns."""
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            _make_rows(n_steps=100, n_replicates=3, param_mu=0.0),
        )
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=100
        )
        assert not df.empty
        assert "wasserstein_mean" in df.columns
        assert "wasserstein_std" in df.columns
        assert "n_replicates" in df.columns
        for key in grid:
            assert key in df.columns

    def test_one_row_per_variant(self, tmp_path):
        """There must be exactly one summary row per variant in the grid."""
        stems = [
            "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            "sensitivity_k=100__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            "sensitivity_k=50__perturbation_scale=0.8__tol_init_multiplier=1.0.csv",
            "sensitivity_k=100__perturbation_scale=0.8__tol_init_multiplier=1.0.csv",
        ]
        for stem in stems:
            _write_variant_csv(
                tmp_path / stem,
                _make_rows(n_steps=100, n_replicates=2, param_mu=0.0),
            )
        grid = {"k": [50, 100], "perturbation_scale": [0.4, 0.8], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=100
        )
        assert len(df) == 4

    def test_wasserstein_is_finite_for_valid_data(self, tmp_path):
        """Wasserstein mean must be a finite, non-negative number for valid CSVs."""
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            _make_rows(n_steps=100, n_replicates=3, param_mu=0.0),
        )
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=100
        )
        assert math.isfinite(df["wasserstein_mean"].iloc[0])
        assert df["wasserstein_mean"].iloc[0] >= 0.0

    def test_missing_variant_csv_produces_nan_row(self, tmp_path):
        """A grid entry with no CSV file must yield a NaN row, not crash."""
        # Write only one of two expected CSVs
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            _make_rows(n_steps=100, n_replicates=2, param_mu=0.0),
        )
        grid = {"k": [50, 100], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=100
        )
        assert len(df) == 2
        nan_row = df[df["k"].astype(str) == "100"]
        assert not nan_row.empty
        assert math.isnan(nan_row["wasserstein_mean"].iloc[0])

    def test_tol_init_variants_give_comparable_wasserstein(self, tmp_path):
        """Critical: variants differing only in tol_init but producing the same
        posterior samples should yield similar Wasserstein scores.

        This is the regression test for the old 'mean final tolerance' metric:
        that metric would make tol_init=0.5 look much better than tol_init=5.0
        even if both converge to the same posterior, because low tol_init
        gives a lower absolute final ε. Wasserstein is agnostic to starting
        conditions.
        """
        true_mu = 0.0
        rng = np.random.default_rng(42)
        # Both variants: posterior samples tightly around true_mu=0
        for tol_mult in (0.5, 5.0):
            _write_variant_csv(
                tmp_path / f"sensitivity_k=50__perturbation_scale=0.8__tol_init_multiplier={tol_mult}.csv",
                _make_rows(
                    n_steps=100, n_replicates=3, param_mu=true_mu,
                    method=f"async_propulate_abc__k=50__perturbation_scale=0.8__tol_init_multiplier={tol_mult}",
                    noise=0.01, rng=rng,
                ),
            )
        grid = {"k": [50], "perturbation_scale": [0.8], "tol_init_multiplier": [0.5, 5.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": true_mu}, max_simulations=100
        )
        # Both variants converge to same posterior → similar Wasserstein
        wass = df.set_index(df["tol_init_multiplier"].astype(str))["wasserstein_mean"]
        assert abs(wass["0.5"] - wass["5.0"]) < 0.1, (
            f"Expected similar Wasserstein for both tol_init variants, "
            f"got {wass['0.5']:.4f} vs {wass['5.0']:.4f}"
        )

    def test_per_replicate_std_is_nonzero_with_noise(self, tmp_path):
        """With noisy posterior samples across replicates, std must be > 0."""
        rows = _make_rows(n_steps=100, n_replicates=5, param_mu=0.0, noise=0.5,
                          rng=np.random.default_rng(7))
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            rows,
        )
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=100
        )
        assert df["wasserstein_std"].iloc[0] > 0.0

    def test_n_replicates_column_reflects_actual_count(self, tmp_path):
        """n_replicates column must match the actual number of replicates in the CSV."""
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            _make_rows(n_steps=50, n_replicates=4, param_mu=0.0),
        )
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=50
        )
        assert df["n_replicates"].iloc[0] == 4


class TestBudgetKeyedTailWindow:
    """The tail window must be keyed by simulation budget (step), not row count."""

    def test_tail_window_uses_step_not_row_count(self, tmp_path):
        """Two variants with the same row count but different step ranges must
        yield tail windows anchored to the step range, not the total row count.

        Variant A: steps 1..100,  true_mu posterior
        Variant B: steps 1..1000, true_mu posterior for steps > 900, far-off for early steps

        Both have same row count but variant B's tail window (steps 900..1000)
        should still capture the converged region.
        """
        true_mu = 0.0
        far_mu = 10.0

        # Variant A: 100 rows, steps 1-100, all at true_mu
        rows_a = _make_rows(n_steps=100, n_replicates=1, param_mu=true_mu,
                            method="async_propulate_abc__k=50__perturbation_scale=0.4__tol_init_multiplier=0.5")
        # Variant B: 100 rows, steps 1-100 range (simulating same budget)
        # but early rows at far_mu, late rows at true_mu
        rows_b = []
        for step in range(1, 101):
            param_val = true_mu if step >= 90 else far_mu
            rows_b.append({
                "method": "async_propulate_abc__k=50__perturbation_scale=0.4__tol_init_multiplier=5.0",
                "replicate": 0, "seed": 0, "step": step,
                "param_mu": param_val, "loss": 0.1, "weight": 1.0,
                "tolerance": 1.0, "wall_time": float(step),
                "worker_id": "", "sim_start_time": "", "sim_end_time": "",
                "generation": "", "record_kind": "", "time_semantics": "",
                "attempt_count": step,
            })

        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=0.5.csv",
            rows_a,
        )
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=5.0.csv",
            rows_b,
        )
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [0.5, 5.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": true_mu}, max_simulations=100, tail_fraction=0.1
        )
        # Both should have low wasserstein because tail window (steps 90-100) captures true_mu
        for _, row in df.iterrows():
            assert row["wasserstein_mean"] < 1.0, (
                f"tol_init_multiplier={row['tol_init_multiplier']}: "
                f"expected wasserstein < 1.0, got {row['wasserstein_mean']:.4f}"
            )

    def test_tail_window_respects_max_simulations_parameter(self, tmp_path):
        """Rows with step > max_simulations are excluded from the tail window."""
        # Write rows for steps 1-200, but max_simulations=100
        # Steps 91-100 are at true_mu, steps 101-200 are at far_mu
        rows = []
        for step in range(1, 201):
            rows.append({
                "method": "async_propulate_abc__k=50__perturbation_scale=0.4__tol_init_multiplier=1.0",
                "replicate": 0, "seed": 0, "step": step,
                "param_mu": 0.0 if 91 <= step <= 100 else 10.0,
                "loss": 0.1, "weight": 1.0, "tolerance": 1.0, "wall_time": float(step),
                "worker_id": "", "sim_start_time": "", "sim_end_time": "",
                "generation": "", "record_kind": "", "time_semantics": "",
                "attempt_count": step,
            })
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            rows,
        )
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=100, tail_fraction=0.1
        )
        # Tail window = steps 90-100: all true_mu → low wasserstein
        assert df["wasserstein_mean"].iloc[0] < 1.0

    def test_few_accepted_particles_gives_estimate_not_crash(self, tmp_path):
        """Even a single particle in the tail window must return a finite value."""
        rows = [{
            "method": "async_propulate_abc__k=50__perturbation_scale=0.4__tol_init_multiplier=1.0",
            "replicate": 0, "seed": 0, "step": 99,
            "param_mu": 0.1, "loss": 0.05, "weight": 1.0,
            "tolerance": 0.5, "wall_time": 1.0,
            "worker_id": "", "sim_start_time": "", "sim_end_time": "",
            "generation": "", "record_kind": "", "time_semantics": "",
            "attempt_count": 99,
        }]
        _write_variant_csv(
            tmp_path / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
            rows,
        )
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        df = compute_sensitivity_quality_summary(
            tmp_path, grid, true_params={"mu": 0.0}, max_simulations=100
        )
        assert math.isfinite(df["wasserstein_mean"].iloc[0])
