"""Tests for Phase 5: plotting module (export + common)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make async_abc importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from matplotlib.figure import Figure

from async_abc.analysis import posterior_quality_curve, tolerance_over_wall_time
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord
import async_abc.plotting.export as export_mod
from async_abc.plotting.export import save_figure, get_git_hash
from async_abc.plotting.common import (
    corner_plot,
    gantt_plot,
    posterior_quality_plot,
    posterior_plot,
    quality_vs_time_plot,
    sensitivity_heatmap,
    scaling_plot,
    threshold_summary_plot,
    tolerance_trajectory_plot,
    archive_evolution_plot,
    compute_wasserstein,
)
from async_abc.plotting.reporters import (
    _compute_idle_fraction,
    _final_population,
    _parse_variant_stem,
    plot_attempts_to_target_summary,
    plot_corner,
    plot_idle_fraction,
    plot_idle_fraction_comparison,
    plot_quality_vs_attempt_budget,
    plot_quality_vs_posterior_samples,
    plot_quality_vs_time,
    plot_quality_vs_wall_time_diagnostic,
    plot_sensitivity_summary,
    plot_throughput_over_time,
    plot_time_to_target_summary,
    plot_tolerance_trajectory,
    plot_worker_gantt,
)
from async_abc.plotting.common import (
    idle_fraction_plot,
    idle_fraction_comparison_plot,
    posterior_comparison_plot,
    throughput_over_time_plot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fig():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _multi_param_records():
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=i + 1,
            params={"x": 0.1 + 0.1 * i, "y": 1.0 - 0.1 * i},
            loss=0.2 * (i + 1),
            weight=1.0 / 4.0,
            tolerance=1.0,
            wall_time=0.1 * (i + 1),
        )
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# export.py tests
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_export_creates_pdf_and_png(self, tmp_path):
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        save_figure(fig, stem)
        assert (tmp_path / "test_fig.pdf").exists()
        assert (tmp_path / "test_fig.png").exists()

    def test_export_creates_data_csv(self, tmp_path):
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        data = {"x": [1, 2], "y": [3, 4]}
        save_figure(fig, stem, data=data)
        csv_path = tmp_path / "test_fig_data.csv"
        assert csv_path.exists()
        rows = _read_csv(csv_path)
        assert len(rows) == 2
        assert set(rows[0].keys()) == {"x", "y"}

    def test_export_creates_meta_json(self, tmp_path):
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        save_figure(fig, stem)
        meta_path = tmp_path / "test_fig_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "git_hash" in meta
        assert "timestamp" in meta

    def test_export_no_data_csv_when_none(self, tmp_path):
        """No _data.csv should be created when data=None."""
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        save_figure(fig, stem, data=None)
        assert not (tmp_path / "test_fig_data.csv").exists()

    def test_export_uses_agg_fallback_when_external_tools_unavailable(self, tmp_path, monkeypatch):
        """When pdftoppm/convert are unavailable, the Agg fallback should create the PNG."""
        fig = _make_fig()
        stem = tmp_path / "fallback_fig"
        monkeypatch.setattr(export_mod, "_save_png_from_pdf", lambda *args, **kwargs: False)

        save_figure(fig, stem)

        assert (tmp_path / "fallback_fig.pdf").exists()
        assert (tmp_path / "fallback_fig.png").exists()
        meta = json.loads((tmp_path / "fallback_fig_meta.json").read_text())
        assert meta["png"] is not None


class TestGetGitHash:
    def test_returns_string(self):
        h = get_git_hash()
        assert isinstance(h, str)
        assert len(h) > 0


# ---------------------------------------------------------------------------
# common.py tests
# ---------------------------------------------------------------------------

class TestPosteriorPlot:
    def test_posterior_plot_saves_files(self, tmp_path):
        samples = np.random.default_rng(0).normal(0, 1, size=(200,))
        posterior_plot(samples, param_name="mu", path_stem=tmp_path / "posterior")
        assert (tmp_path / "posterior.pdf").exists()
        assert (tmp_path / "posterior.png").exists()

    def test_posterior_plot_returns_paths(self, tmp_path):
        samples = np.random.default_rng(1).normal(size=(100,))
        paths = posterior_plot(samples, param_name="theta", path_stem=tmp_path / "post")
        assert "pdf" in paths and "png" in paths


class TestScalingPlot:
    def test_scaling_plot_saves_files(self, tmp_path):
        throughput = {1: 10.0, 2: 18.0, 4: 32.0, 8: 55.0}
        scaling_plot(throughput, path_stem=tmp_path / "scaling")
        assert (tmp_path / "scaling.pdf").exists()
        assert (tmp_path / "scaling.png").exists()

    def test_scaling_plot_efficiency_curve(self, tmp_path):
        """efficiency[n] == throughput[n] / (n * throughput[1])"""
        throughput = {1: 10.0, 2: 18.0, 4: 32.0}
        scaling_plot(throughput, path_stem=tmp_path / "scaling")
        rows = _read_csv(tmp_path / "scaling_data.csv")
        assert set(rows[0].keys()) >= {"n_workers", "throughput", "efficiency"}
        t1 = throughput[1]
        for row in rows:
            n = float(row["n_workers"])
            t = float(row["throughput"])
            eff = float(row["efficiency"])
            expected = t / (n * t1)
            assert abs(eff - expected) < 1e-9


class TestArchiveEvolutionPlot:
    def test_archive_evolution_plot_saves_files(self, tmp_path):
        sim_counts = np.array([0, 100, 200, 300, 400, 500])
        tolerances = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.8])
        archive_evolution_plot(sim_counts, tolerances, path_stem=tmp_path / "evolution")
        assert (tmp_path / "evolution.pdf").exists()
        assert (tmp_path / "evolution.png").exists()

    def test_archive_evolution_plot_data_csv(self, tmp_path):
        sim_counts = np.array([0, 100, 200])
        tolerances = np.array([5.0, 3.0, 1.5])
        archive_evolution_plot(sim_counts, tolerances, path_stem=tmp_path / "evo")
        rows = _read_csv(tmp_path / "evo_data.csv")
        assert set(rows[0].keys()) >= {"sim_count", "tolerance"}


class TestSensitivityHeatmap:
    def test_sensitivity_heatmap_supports_facets(self, tmp_path):
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ])
        paths = sensitivity_heatmap(
            data,
            row_labels=["50", "100"],
            col_labels=["0.4", "0.8"],
            path_stem=tmp_path / "sensitivity_facet",
            facet_labels=["0.5x", "2.0x"],
        )
        assert "pdf" in paths and "png" in paths and "csv" in paths

    def test_sensitivity_heatmap_handles_all_nan_data(self, tmp_path):
        data = np.full((2, 2, 2), np.nan)
        paths = sensitivity_heatmap(
            data,
            row_labels=["50", "100"],
            col_labels=["0.4", "0.8"],
            path_stem=tmp_path / "sensitivity_nan",
            facet_labels=["0.5x", "2.0x"],
        )
        assert "pdf" in paths and "png" in paths and "csv" in paths


class TestPhase3PlotFunctions:
    def test_gantt_plot_returns_figure(self, abc_smc_records):
        fig = gantt_plot(abc_smc_records)
        assert isinstance(fig, Figure)

    def test_quality_vs_time_plot_returns_figure(self, sample_records):
        df = posterior_quality_curve(sample_records, {"mu": 0.0}, axis_kind="wall_time", archive_size=20)
        fig = quality_vs_time_plot(df)
        assert isinstance(fig, Figure)

    def test_posterior_quality_plot_supports_attempt_budget(self, sample_records):
        df = posterior_quality_curve(sample_records, {"mu": 0.0}, axis_kind="attempt_budget", archive_size=20)
        fig = posterior_quality_plot(df, axis_kind="attempt_budget")
        assert isinstance(fig, Figure)

    def test_posterior_quality_plot_handles_generation_snapshot_duplicates(self, abc_smc_records):
        df = posterior_quality_curve(abc_smc_records, {"mu": 0.0}, axis_kind="wall_time")
        fig = posterior_quality_plot(df, axis_kind="wall_time")
        assert isinstance(fig, Figure)

    def test_threshold_summary_plot_returns_figure(self, sample_records):
        summary_df = (
            posterior_quality_curve(sample_records, {"mu": 0.0}, axis_kind="wall_time", archive_size=20)
            .head(2)
            .rename(columns={"axis_value": "axis_value_to_threshold"})
        )
        fig = threshold_summary_plot(summary_df, axis_kind="wall_time")
        assert isinstance(fig, Figure)

    def test_corner_plot_returns_figure(self):
        fig = corner_plot(_multi_param_records(), param_names=["x", "y"], true_params={"x": 0.0, "y": 1.0})
        assert isinstance(fig, Figure)

    def test_tolerance_trajectory_plot_returns_figure(self, sample_records):
        df = tolerance_over_wall_time(sample_records)
        fig = tolerance_trajectory_plot(df)
        assert isinstance(fig, Figure)


class TestPhase3Reporters:
    def test_plot_worker_gantt_exports_files(self, tmp_path, abc_smc_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_worker_gantt(abc_smc_records, output_dir)
        assert (output_dir.plots / "worker_gantt.pdf").exists()
        assert (output_dir.plots / "worker_gantt.png").exists()
        assert (output_dir.plots / "worker_gantt_data.csv").exists()
        assert (output_dir.plots / "worker_gantt_meta.json").exists()

    def test_plot_quality_vs_time_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_time(sample_records, {"mu": 0.0}, [10, 50], output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_time.pdf").exists()
        assert (output_dir.plots / "quality_vs_time.png").exists()
        assert (output_dir.plots / "quality_vs_time_data.csv").exists()
        assert (output_dir.plots / "quality_vs_time_meta.json").exists()
        assert (output_dir.plots / "quality_vs_wall_time_diagnostic.pdf").exists()
        meta = json.loads((output_dir.plots / "quality_vs_time_meta.json").read_text())
        assert meta["deprecated_alias_for"] == "quality_vs_wall_time_diagnostic"

    def test_plot_quality_vs_wall_time_diagnostic_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_wall_time_diagnostic(sample_records, {"mu": 0.0}, output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_wall_time_diagnostic.pdf").exists()
        meta = json.loads((output_dir.plots / "quality_vs_wall_time_diagnostic_meta.json").read_text())
        assert meta["plot_name"] == "quality_vs_wall_time_diagnostic"
        assert meta["axis_kind"] == "wall_time"

    def test_plot_quality_vs_posterior_samples_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_posterior_samples(sample_records, {"mu": 0.0}, output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_posterior_samples.pdf").exists()

    def test_plot_quality_vs_attempt_budget_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_attempt_budget(sample_records, {"mu": 0.0}, output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_attempt_budget.pdf").exists()

    def test_plot_time_to_target_summary_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_time_to_target_summary(sample_records, {"mu": 0.0}, 10.0, output_dir, archive_size=20)
        assert (output_dir.plots / "time_to_target_summary.pdf").exists()

    def test_plot_attempts_to_target_summary_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_attempts_to_target_summary(sample_records, {"mu": 0.0}, 10.0, output_dir, archive_size=20)
        assert (output_dir.plots / "attempts_to_target_summary.pdf").exists()

    def test_plot_corner_exports_files(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_corner(_multi_param_records(), ["x", "y"], output_dir, true_params={"x": 0.0, "y": 1.0})
        assert (output_dir.plots / "corner.pdf").exists()
        assert (output_dir.plots / "corner.png").exists()
        assert (output_dir.plots / "corner_data.csv").exists()
        assert (output_dir.plots / "corner_meta.json").exists()

    def test_plot_tolerance_trajectory_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_tolerance_trajectory(sample_records, output_dir)
        assert (output_dir.plots / "tolerance_trajectory.pdf").exists()
        assert (output_dir.plots / "tolerance_trajectory.png").exists()
        assert (output_dir.plots / "tolerance_trajectory_data.csv").exists()
        assert (output_dir.plots / "tolerance_trajectory_meta.json").exists()

    def test_plot_sensitivity_summary_exports_files(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        data_dir = output_dir.data
        rows = [
            {"tolerance": "2.0"},
            {"tolerance": "1.5"},
            {"tolerance": "1.0"},
        ]
        for stem in (
            "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=0.5.csv",
            "sensitivity_k=100__perturbation_scale=0.4__tol_init_multiplier=0.5.csv",
            "sensitivity_k=50__perturbation_scale=0.8__tol_init_multiplier=2.0.csv",
            "sensitivity_k=100__perturbation_scale=0.8__tol_init_multiplier=2.0.csv",
        ):
            with open(data_dir / stem, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["tolerance"])
                writer.writeheader()
                writer.writerows(rows)
        grid = {
            "k": [50, 100],
            "perturbation_scale": [0.4, 0.8],
            "tol_init_multiplier": [0.5, 2.0],
        }
        plot_sensitivity_summary(data_dir, grid, output_dir)
        assert (output_dir.plots / "sensitivity_heatmap.pdf").exists()
        assert (output_dir.plots / "sensitivity_heatmap.png").exists()
        assert (output_dir.plots / "sensitivity_heatmap_data.csv").exists()


class TestSensitivityVariantParsing:
    def test_parses_new_variant_stem(self):
        variant = _parse_variant_stem(
            "k=50__perturbation_scale=0.4__scheduler_type=acceptance_rate__tol_init_multiplier=0.5",
            ["k", "perturbation_scale", "scheduler_type", "tol_init_multiplier"],
        )
        assert variant == {
            "k": 50,
            "perturbation_scale": 0.4,
            "scheduler_type": "acceptance_rate",
            "tol_init_multiplier": 0.5,
        }

    def test_parses_legacy_variant_stem(self):
        variant = _parse_variant_stem(
            "k50_perturbation_scale0.4_scheduler_typeacceptance_rate_tol_init_multiplier0.5",
            ["k", "perturbation_scale", "scheduler_type", "tol_init_multiplier"],
        )
        assert variant == {
            "k": 50,
            "perturbation_scale": 0.4,
            "scheduler_type": "acceptance_rate",
            "tol_init_multiplier": 0.5,
        }


class TestWassersteinMetric:
    def test_wasserstein_metric_returns_float(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(1, 1, 200)
        d = compute_wasserstein(a, b)
        assert isinstance(d, float)

    def test_wasserstein_same_distribution_near_zero(self):
        a = np.random.default_rng(7).normal(0, 1, 1000)
        d = compute_wasserstein(a, a)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_wasserstein_positive_for_different(self):
        rng = np.random.default_rng(99)
        a = rng.normal(0, 1, 500)
        b = rng.normal(5, 1, 500)
        d = compute_wasserstein(a, b)
        assert d > 0


# ---------------------------------------------------------------------------
# _final_population per-method fix
# ---------------------------------------------------------------------------

class TestFinalPopulationPerMethod:
    def test_returns_particles_from_all_methods(self):
        """Each method's lowest-tolerance particles should be included."""
        records = [
            ParticleRecord(method="method_a", replicate=0, seed=1, step=1,
                           params={"x": 1.0}, loss=0.5, tolerance=2.0, wall_time=1.0),
            ParticleRecord(method="method_a", replicate=0, seed=1, step=2,
                           params={"x": 1.1}, loss=0.3, tolerance=1.0, wall_time=2.0),
            ParticleRecord(method="method_b", replicate=0, seed=1, step=1,
                           params={"x": 2.0}, loss=0.8, tolerance=5.0, wall_time=1.0),
            ParticleRecord(method="method_b", replicate=0, seed=1, step=2,
                           params={"x": 2.1}, loss=0.6, tolerance=3.0, wall_time=2.0),
        ]
        final = _final_population(records)
        methods = {r.method for r in final}
        assert methods == {"method_a", "method_b"}

    def test_selects_per_method_minimum_tolerance(self):
        records = [
            ParticleRecord(method="A", replicate=0, seed=1, step=1,
                           params={"x": 1.0}, loss=0.5, tolerance=1.0, wall_time=1.0),
            ParticleRecord(method="A", replicate=0, seed=1, step=2,
                           params={"x": 1.1}, loss=0.3, tolerance=2.0, wall_time=2.0),
            ParticleRecord(method="B", replicate=0, seed=1, step=1,
                           params={"x": 2.0}, loss=0.8, tolerance=10.0, wall_time=1.0),
        ]
        final = _final_population(records)
        assert len(final) == 2  # method A tol=1.0 + method B tol=10.0
        assert {r.method for r in final} == {"A", "B"}
        assert {r.tolerance for r in final} == {1.0, 10.0}


# ---------------------------------------------------------------------------
# Posterior comparison plot
# ---------------------------------------------------------------------------

class TestPosteriorComparisonPlot:
    def test_saves_files(self, tmp_path):
        rng = np.random.default_rng(42)
        method_samples = {
            "method_a": rng.normal(0, 1, 100),
            "method_b": rng.normal(1, 1, 100),
        }
        posterior_comparison_plot(method_samples, param_name="mu", path_stem=tmp_path / "post_cmp")
        assert (tmp_path / "post_cmp.pdf").exists()
        assert (tmp_path / "post_cmp.png").exists()


# ---------------------------------------------------------------------------
# Corner plot with method labels
# ---------------------------------------------------------------------------

class TestCornerPlotMethodLabels:
    def test_corner_plot_with_methods(self):
        records = [
            ParticleRecord(method="A", replicate=0, seed=1, step=i,
                           params={"x": float(i), "y": float(i) * 2}, loss=0.1, wall_time=0.1)
            for i in range(10)
        ] + [
            ParticleRecord(method="B", replicate=0, seed=1, step=i,
                           params={"x": float(i) + 5, "y": float(i) * 3}, loss=0.1, wall_time=0.1)
            for i in range(10)
        ]
        fig = corner_plot(records, param_names=["x", "y"], method_labels=["A", "B"])
        assert fig is not None


# ---------------------------------------------------------------------------
# Idle fraction / throughput / comparison plots
# ---------------------------------------------------------------------------

def _heterogeneity_records():
    """Synthetic records mimicking runtime_heterogeneity output."""
    records = []
    for sigma in [0.0, 0.5, 1.0]:
        for rep in range(2):
            for step in range(5):
                records.append(ParticleRecord(
                    method=f"async_propulate_abc__sigma{sigma}",
                    replicate=rep, seed=42, step=step,
                    params={"x": float(step)}, loss=float(step) * 0.1,
                    tolerance=1.0,
                    wall_time=0.1 * step,
                    worker_id=str(step % 2),
                    sim_start_time=0.1 * step,
                    sim_end_time=0.1 * step + 0.08,
                ))
    return records


class TestIdleFractionPlot:
    def test_idle_fraction_plot_saves_files(self, tmp_path):
        idle_fraction_plot([0.0, 0.5, 1.0], [0.1, 0.3, 0.5], tmp_path / "idle")
        assert (tmp_path / "idle.pdf").exists()

    def test_plot_idle_fraction_reporter(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction(_heterogeneity_records(), output_dir)
        assert (output_dir.plots / "idle_fraction.pdf").exists()


class TestThroughputOverTimePlot:
    def test_throughput_over_time_plot_saves_files(self, tmp_path):
        time_bins = {"σ=0.0": np.array([0.5, 1.5, 2.5])}
        throughput_bins = {"σ=0.0": np.array([5.0, 3.0, 2.0])}
        throughput_over_time_plot(time_bins, throughput_bins, tmp_path / "throughput")
        assert (tmp_path / "throughput.pdf").exists()

    def test_plot_throughput_over_time_reporter(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_throughput_over_time(_heterogeneity_records(), output_dir)
        assert (output_dir.plots / "throughput_over_time.pdf").exists()


class TestIdleFractionComparisonPlot:
    def test_idle_fraction_comparison_plot_saves_files(self, tmp_path):
        idle_by_sigma = {0.0: [0.1, 0.15], 0.5: [0.3, 0.35], 1.0: [0.5, 0.55]}
        idle_fraction_comparison_plot([0.0, 0.5, 1.0], idle_by_sigma, tmp_path / "idle_cmp")
        assert (tmp_path / "idle_cmp.pdf").exists()

    def test_plot_idle_fraction_comparison_reporter(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction_comparison(_heterogeneity_records(), output_dir)
        assert (output_dir.plots / "idle_fraction_comparison.pdf").exists()


class TestComputeIdleFraction:
    def test_returns_per_method_per_replicate(self):
        records = _heterogeneity_records()
        idle = _compute_idle_fraction(records)
        assert "async_propulate_abc__sigma0.0" in idle
        assert 0 in idle["async_propulate_abc__sigma0.0"]
        assert 0 <= idle["async_propulate_abc__sigma0.0"][0] <= 1


# ---------------------------------------------------------------------------
# PNG Agg fallback
# ---------------------------------------------------------------------------

class TestPngAggFallback:
    def test_png_created_when_external_tools_unavailable(self, tmp_path, monkeypatch):
        """Even without pdftoppm/convert, PNG should be created via Agg fallback."""
        fig = _make_fig()
        monkeypatch.setattr(export_mod, "_save_png_from_pdf", lambda *a, **kw: False)
        stem = tmp_path / "fallback_fig"
        result = save_figure(fig, stem)
        assert (tmp_path / "fallback_fig.png").exists()
        meta = json.loads((tmp_path / "fallback_fig_meta.json").read_text())
        assert meta["png"] is not None
